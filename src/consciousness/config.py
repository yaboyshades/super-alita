from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.core import yaml_utils


@dataclass(slots=True)
class RedisConfig:
    """Redis connection configuration."""

    url: str
    max_connections: int = 100

    @property
    def host(self) -> str:
        parsed = urlparse(self.url)
        return parsed.hostname or "localhost"

    @property
    def port(self) -> int:
        parsed = urlparse(self.url)
        return parsed.port or 6379

    @property
    def db(self) -> int:
        parsed = urlparse(self.url)
        try:
            db_str = (parsed.path or "/0").strip("/")
            return int(db_str or 0)
        except ValueError:
            return 0


@dataclass(slots=True)
class EventBusConfig:
    max_message_size: str = "10MB"
    retention_hours: int = 24
    constitutional_validation: bool = True


@dataclass(slots=True)
class MemoryConfig:
    neural_federation_enabled: bool = True
    cross_component_indexing: bool = True
    embedding_model: str = "text-embedding-3-large"


@dataclass(slots=True)
class ConstitutionalConfig:
    compliance_threshold: float = 0.75
    auto_remediation: bool = True
    evolution_enabled: bool = True


@dataclass(slots=True)
class InfrastructureConfig:
    redis: RedisConfig
    eventbus: EventBusConfig = field(default_factory=EventBusConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    constitutional: ConstitutionalConfig = field(
        default_factory=ConstitutionalConfig
    )


@dataclass(slots=True)
class ComponentSettings:
    name: str
    process_command: tuple[str, ...]
    healthcheck_endpoint: str | None = None
    restart_on_failure: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NeuralPathwayConfig:
    type: str = "bidirectional"
    strength: float = 0.8
    adaptation_rate: float = 0.1


@dataclass(slots=True)
class ConsciousnessLearningConfig:
    cross_component_enabled: bool = True
    pattern_extraction_interval: int = 300
    knowledge_synthesis_enabled: bool = True


@dataclass(slots=True)
class ConsciousnessPredictionConfig:
    architectural_forecasting: bool = True
    prediction_horizon_days: int = 90
    confidence_threshold: float = 0.7


@dataclass(slots=True)
class SelfModificationConfig:
    enabled: bool = True
    safety_checks: tuple[str, ...] = (
        "constitutional",
        "performance",
        "rollback",
    )
    modification_approval_threshold: float = 0.9


@dataclass(slots=True)
class ConsciousnessRuntimeConfig:
    neural_pathways: dict[str, NeuralPathwayConfig] = field(
        default_factory=dict
    )
    learning: ConsciousnessLearningConfig = field(
        default_factory=ConsciousnessLearningConfig
    )
    prediction: ConsciousnessPredictionConfig = field(
        default_factory=ConsciousnessPredictionConfig
    )
    self_modification: SelfModificationConfig = field(
        default_factory=SelfModificationConfig
    )


@dataclass(slots=True)
class TribalKnowledgeExtractorConfig:
    enabled: bool = False
    integration_depth: str = "core"
    integration_points: dict[str, bool] = field(default_factory=dict)
    confidence_threshold: float = 0.8
    registry_path: str | None = None


@dataclass(slots=True)
class LivingArchitectureEngineConfig:
    enabled: bool = False
    runtime_host: str | None = None
    metrics_endpoints: tuple[str, ...] = ()
    reality_gap_threshold: float = 0.15
    bidirectional_sync: bool = False


@dataclass(slots=True)
class UniversalPatternMinerConfig:
    enabled: bool = False
    integration_points: dict[str, bool] = field(default_factory=dict)
    mining_strategies: tuple[str, ...] = ()
    universality_threshold: float = 0.9
    context_adaptation_enabled: bool = False


@dataclass(slots=True)
class ArchitecturalTimeTravelConfig:
    enabled: bool = False
    prediction_horizons: tuple[str, ...] = ("3_months", "1_year")
    debt_forecasting: bool = False
    pressure_point_detection: bool = False


@dataclass(slots=True)
class ConstitutionalEvolutionConfig:
    enabled: bool = False
    integration_points: dict[str, bool] = field(default_factory=dict)
    acceptance_threshold: float = 0.6
    evolution_iterations: int = 500
    evidence_strength_required: float = 0.95
    constitutional_path: str = ".github/CONSTITUTION.md"
    meta_constitutional_review: bool = True
    backward_compatibility_check: bool = True


@dataclass(slots=True)
class ArchitecturalConsciousnessConfig:
    enabled: bool = False
    consciousness_threshold: float = 0.7
    emergent_intelligence: bool = False
    temporal_reasoning: bool = False


@dataclass(slots=True)
class ArchitecturalImmuneSystemConfig:
    enabled: bool = False
    health_monitoring: str = "manual"
    infection_detection: str = "none"
    immune_responses: tuple[str, ...] = ()
    immunity_development: bool = False
    auto_healing: bool = False


@dataclass(slots=True)
class PhilosophicalArchitectureConfig:
    enabled: bool = False
    socratic_depth: str = "surface"


@dataclass(slots=True)
class AdvancedCapabilitiesConfig:
    tribal_knowledge_extractor: TribalKnowledgeExtractorConfig = field(
        default_factory=TribalKnowledgeExtractorConfig
    )
    living_architecture_engine: LivingArchitectureEngineConfig = field(
        default_factory=LivingArchitectureEngineConfig
    )
    universal_pattern_miner: UniversalPatternMinerConfig = field(
        default_factory=UniversalPatternMinerConfig
    )
    architectural_time_travel: ArchitecturalTimeTravelConfig = field(
        default_factory=ArchitecturalTimeTravelConfig
    )
    constitutional_evolution: ConstitutionalEvolutionConfig = field(
        default_factory=ConstitutionalEvolutionConfig
    )
    architectural_consciousness: ArchitecturalConsciousnessConfig = field(
        default_factory=ArchitecturalConsciousnessConfig
    )
    architectural_immune_system: ArchitecturalImmuneSystemConfig = field(
        default_factory=ArchitecturalImmuneSystemConfig
    )
    philosophical_architecture: PhilosophicalArchitectureConfig = field(
        default_factory=PhilosophicalArchitectureConfig
    )

    def any_enabled(self) -> bool:
        return any(
            (
                self.tribal_knowledge_extractor.enabled,
                self.living_architecture_engine.enabled,
                self.universal_pattern_miner.enabled,
                self.architectural_time_travel.enabled,
                self.constitutional_evolution.enabled,
                self.architectural_consciousness.enabled,
                self.architectural_immune_system.enabled,
                self.philosophical_architecture.enabled,
            )
        )

    def enabled_capability_names(self) -> Iterable[str]:
        capability_map = {
            "tribal_knowledge_extractor": self.tribal_knowledge_extractor.enabled,
            "living_architecture_engine": self.living_architecture_engine.enabled,
            "universal_pattern_miner": self.universal_pattern_miner.enabled,
            "architectural_time_travel": self.architectural_time_travel.enabled,
            "constitutional_evolution": self.constitutional_evolution.enabled,
            "architectural_consciousness": self.architectural_consciousness.enabled,
            "architectural_immune_system": self.architectural_immune_system.enabled,
            "philosophical_architecture": self.philosophical_architecture.enabled,
        }
        return (name for name, enabled in capability_map.items() if enabled)


@dataclass(slots=True)
class BootstrapConfig:
    bootstrap_timeout_seconds: int
    consciousness_emergence_threshold: float
    infrastructure: InfrastructureConfig
    components: dict[str, ComponentSettings]
    consciousness: ConsciousnessRuntimeConfig
    advanced_capabilities: AdvancedCapabilitiesConfig | None = None

    def component_names(self) -> tuple[str, ...]:
        return tuple(self.components.keys())

    def advanced_enabled(self) -> bool:
        return bool(
            self.advanced_capabilities
            and self.advanced_capabilities.any_enabled()
        )


def _component_from(name: str, data: dict[str, Any]) -> ComponentSettings:
    cmd = tuple(str(item) for item in data.get("process_command", []))
    if not cmd:
        raise ValueError(
            f"Component '{name}' requires a non-empty process_command list"
        )
    settings = ComponentSettings(
        name=name,
        process_command=cmd,
        healthcheck_endpoint=data.get("healthcheck_endpoint"),
        restart_on_failure=bool(data.get("restart_on_failure", True)),
        options={
            k: v
            for k, v in data.items()
            if k
            not in {
                "process_command",
                "healthcheck_endpoint",
                "restart_on_failure",
            }
        },
    )
    return settings


def _pathway_from(data: dict[str, Any]) -> NeuralPathwayConfig:
    return NeuralPathwayConfig(
        type=str(data.get("type", "bidirectional")),
        strength=float(data.get("strength", 0.8)),
        adaptation_rate=float(data.get("adaptation_rate", 0.1)),
    )


def _tuple_of_str(values: Any) -> tuple[str, ...]:
    if isinstance(values, list | tuple):
        return tuple(str(item) for item in values)
    if values is None:
        return ()
    return (str(values),)


def _bool_map(data: Any) -> dict[str, bool]:
    if not isinstance(data, dict):
        return {}
    return {str(key): bool(value) for key, value in data.items()}


def _tribal_from(data: dict[str, Any]) -> TribalKnowledgeExtractorConfig:
    return TribalKnowledgeExtractorConfig(
        enabled=bool(data.get("enabled", False)),
        integration_depth=str(data.get("integration_depth", "core")),
        integration_points=_bool_map(data.get("integration_points")),
        confidence_threshold=float(data.get("confidence_threshold", 0.8)),
        registry_path=data.get("registry_path"),
    )


def _living_architecture_from(
    data: dict[str, Any],
) -> LivingArchitectureEngineConfig:
    return LivingArchitectureEngineConfig(
        enabled=bool(data.get("enabled", False)),
        runtime_host=data.get("runtime_host"),
        metrics_endpoints=_tuple_of_str(data.get("metrics_endpoints")),
        reality_gap_threshold=float(data.get("reality_gap_threshold", 0.15)),
        bidirectional_sync=bool(data.get("bidirectional_sync", False)),
    )


def _universal_pattern_from(
    data: dict[str, Any],
) -> UniversalPatternMinerConfig:
    return UniversalPatternMinerConfig(
        enabled=bool(data.get("enabled", False)),
        integration_points=_bool_map(data.get("integration_points")),
        mining_strategies=_tuple_of_str(data.get("mining_strategies")),
        universality_threshold=float(data.get("universality_threshold", 0.9)),
        context_adaptation_enabled=bool(
            data.get("context_adaptation_enabled", False)
        ),
    )


def _time_travel_from(data: dict[str, Any]) -> ArchitecturalTimeTravelConfig:
    return ArchitecturalTimeTravelConfig(
        enabled=bool(data.get("enabled", False)),
        prediction_horizons=_tuple_of_str(data.get("prediction_horizons")),
        debt_forecasting=bool(data.get("debt_forecasting", False)),
        pressure_point_detection=bool(
            data.get("pressure_point_detection", False)
        ),
    )


def _constitutional_evolution_from(
    data: dict[str, Any],
) -> ConstitutionalEvolutionConfig:
    return ConstitutionalEvolutionConfig(
        enabled=bool(data.get("enabled", False)),
        integration_points=_bool_map(data.get("integration_points")),
        acceptance_threshold=float(data.get("acceptance_threshold", 0.6)),
        evolution_iterations=int(data.get("evolution_iterations", 500)),
        evidence_strength_required=float(
            data.get("evidence_strength_required", 0.95)
        ),
        constitutional_path=str(
            data.get("constitutional_path", ".github/CONSTITUTION.md")
        ),
        meta_constitutional_review=bool(
            data.get("meta_constitutional_review", True)
        ),
        backward_compatibility_check=bool(
            data.get("backward_compatibility_check", True)
        ),
    )


def _architectural_consciousness_from(
    data: dict[str, Any],
) -> ArchitecturalConsciousnessConfig:
    return ArchitecturalConsciousnessConfig(
        enabled=bool(data.get("enabled", False)),
        consciousness_threshold=float(
            data.get("consciousness_threshold", 0.7)
        ),
        emergent_intelligence=bool(data.get("emergent_intelligence", False)),
        temporal_reasoning=bool(data.get("temporal_reasoning", False)),
    )


def _architectural_immune_from(
    data: dict[str, Any],
) -> ArchitecturalImmuneSystemConfig:
    return ArchitecturalImmuneSystemConfig(
        enabled=bool(data.get("enabled", False)),
        health_monitoring=str(data.get("health_monitoring", "manual")),
        infection_detection=str(data.get("infection_detection", "none")),
        immune_responses=_tuple_of_str(data.get("immune_responses")),
        immunity_development=bool(data.get("immunity_development", False)),
        auto_healing=bool(data.get("auto_healing", False)),
    )


def _philosophical_architecture_from(
    data: dict[str, Any],
) -> PhilosophicalArchitectureConfig:
    return PhilosophicalArchitectureConfig(
        enabled=bool(data.get("enabled", False)),
        socratic_depth=str(data.get("socratic_depth", "surface")),
    )


def _advanced_from(data: dict[str, Any]) -> AdvancedCapabilitiesConfig:
    return AdvancedCapabilitiesConfig(
        tribal_knowledge_extractor=_tribal_from(
            data.get("tribal_knowledge_extractor", {})
        ),
        living_architecture_engine=_living_architecture_from(
            data.get("living_architecture_engine", {})
        ),
        universal_pattern_miner=_universal_pattern_from(
            data.get("universal_pattern_miner", {})
        ),
        architectural_time_travel=_time_travel_from(
            data.get("architectural_time_travel", {})
        ),
        constitutional_evolution=_constitutional_evolution_from(
            data.get("constitutional_evolution", {})
        ),
        architectural_consciousness=_architectural_consciousness_from(
            data.get("architectural_consciousness", {})
        ),
        architectural_immune_system=_architectural_immune_from(
            data.get("architectural_immune_system", {})
        ),
        philosophical_architecture=_philosophical_architecture_from(
            data.get("philosophical_architecture", {})
        ),
    )


def load_config(path: str | Path) -> BootstrapConfig:
    """Load unified consciousness configuration from YAML."""

    payload = yaml_utils.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or "unified_consciousness" not in payload:
        raise ValueError(
            "Invalid unified consciousness configuration: missing 'unified_consciousness' root"
        )

    root = payload["unified_consciousness"]

    infrastructure = root.get("infrastructure", {})
    redis_cfg = infrastructure.get("redis", {})
    eventbus_cfg = infrastructure.get("eventbus", {})
    memory_cfg = infrastructure.get("memory", {})
    constitutional_cfg = infrastructure.get("constitutional", {})

    infra = InfrastructureConfig(
        redis=RedisConfig(
            url=str(redis_cfg.get("url", "redis://localhost:6379/0")),
            max_connections=int(redis_cfg.get("max_connections", 100)),
        ),
        eventbus=EventBusConfig(
            max_message_size=str(eventbus_cfg.get("max_message_size", "10MB")),
            retention_hours=int(eventbus_cfg.get("retention_hours", 24)),
            constitutional_validation=bool(
                eventbus_cfg.get("constitutional_validation", True)
            ),
        ),
        memory=MemoryConfig(
            neural_federation_enabled=bool(
                memory_cfg.get("neural_federation_enabled", True)
            ),
            cross_component_indexing=bool(
                memory_cfg.get("cross_component_indexing", True)
            ),
            embedding_model=str(
                memory_cfg.get("embedding_model", "text-embedding-3-large")
            ),
        ),
        constitutional=ConstitutionalConfig(
            compliance_threshold=float(
                constitutional_cfg.get("compliance_threshold", 0.75)
            ),
            auto_remediation=bool(
                constitutional_cfg.get("auto_remediation", True)
            ),
            evolution_enabled=bool(
                constitutional_cfg.get("evolution_enabled", True)
            ),
        ),
    )

    components_section = root.get("components", {})
    components: dict[str, ComponentSettings] = {}
    for name, data in components_section.items():
        if not isinstance(data, dict):
            raise ValueError(
                f"Component '{name}' configuration must be a mapping"
            )
        components[name] = _component_from(name, data)

    consciousness_section = root.get("consciousness", {})
    pathways_data = consciousness_section.get("neural_pathways", {})
    neural_pathways = {
        key: _pathway_from(value)
        for key, value in pathways_data.items()
        if isinstance(value, dict)
    }

    learning_cfg = consciousness_section.get("learning", {})
    learning = ConsciousnessLearningConfig(
        cross_component_enabled=bool(
            learning_cfg.get("cross_component_enabled", True)
        ),
        pattern_extraction_interval=int(
            learning_cfg.get("pattern_extraction_interval", 300)
        ),
        knowledge_synthesis_enabled=bool(
            learning_cfg.get("knowledge_synthesis_enabled", True)
        ),
    )

    prediction_cfg = consciousness_section.get("prediction", {})
    prediction = ConsciousnessPredictionConfig(
        architectural_forecasting=bool(
            prediction_cfg.get("architectural_forecasting", True)
        ),
        prediction_horizon_days=int(
            prediction_cfg.get("prediction_horizon_days", 90)
        ),
        confidence_threshold=float(
            prediction_cfg.get("confidence_threshold", 0.7)
        ),
    )

    self_mod_cfg = consciousness_section.get("self_modification", {})
    safety = self_mod_cfg.get(
        "safety_checks", ("constitutional", "performance", "rollback")
    )
    if isinstance(safety, list):
        safety_tuple = tuple(str(item) for item in safety)
    else:
        safety_tuple = ("constitutional", "performance", "rollback")

    self_modification = SelfModificationConfig(
        enabled=bool(self_mod_cfg.get("enabled", True)),
        safety_checks=safety_tuple,
        modification_approval_threshold=float(
            self_mod_cfg.get("modification_approval_threshold", 0.9)
        ),
    )

    consciousness = ConsciousnessRuntimeConfig(
        neural_pathways=neural_pathways,
        learning=learning,
        prediction=prediction,
        self_modification=self_modification,
    )

    advanced_config = None
    advanced_section = root.get("advanced_capabilities")
    if isinstance(advanced_section, dict):
        advanced_config = _advanced_from(advanced_section)

    config = BootstrapConfig(
        bootstrap_timeout_seconds=int(
            root.get("bootstrap_timeout_seconds", 30)
        ),
        consciousness_emergence_threshold=float(
            root.get("consciousness_emergence_threshold", 0.7)
        ),
        infrastructure=infra,
        components=components,
        consciousness=consciousness,
        advanced_capabilities=advanced_config,
    )
    return config
