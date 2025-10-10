from __future__ import annotations

import asyncio
import contextlib
import statistics
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from src.contracts import UnifiedEvent

from .components import ComponentProcess
from .config import NeuralPathwayConfig
from .infrastructure import SystemInfrastructure

BackgroundFactory = Callable[[], Awaitable[None]]


class ConsciousnessOrchestrator:
    """High-level coordinator that assembles unified consciousness."""

    def __init__(
        self,
        infrastructure: SystemInfrastructure,
        components: dict[str, ComponentProcess],
    ) -> None:
        self._infra = infrastructure
        self._components = components
        self._config = infrastructure.config

    async def emerge(self) -> UnifiedConsciousness:
        pathways = await self.form_cross_component_pathways()
        feedback_loops = await self.establish_constitutional_loops()
        reasoning_chains = await self.initialize_reasoning_chains()
        self_modification = await self.enable_safe_self_modification()

        consciousness = UnifiedConsciousness(
            infrastructure=self._infra,
            components=self._components,
            pathways=pathways,
            feedback_loops=feedback_loops,
            reasoning_chains=reasoning_chains,
            self_modification=self_modification,
        )
        await consciousness.bind_eventbus()

        if self._infra.advanced and self._infra.advanced.has_capabilities():
            extra_factories = (
                await self._infra.advanced.prepare_consciousness_extensions(
                    consciousness
                )
            )
            consciousness.add_background_task_factories(extra_factories)

        score = await consciousness.assess_consciousness_emergence()
        threshold = self._config.consciousness_emergence_threshold
        if score < threshold:
            raise ConsciousnessEmergenceFailure(
                f"Consciousness emergence score {score:.3f} below threshold {threshold:.2f}"
            )
        return consciousness

    async def form_cross_component_pathways(self) -> dict[str, dict[str, Any]]:
        pathways: dict[str, dict[str, Any]] = {}
        pathway_configs = self._config.consciousness.neural_pathways
        for pathway_name, cfg in pathway_configs.items():
            source, target = self._derive_endpoints(pathway_name)
            entry = await self.create_pathway(
                source=source,
                target=target,
                pathway_cfg=cfg,
            )
            pathways[pathway_name] = entry
        return pathways

    async def create_pathway(
        self,
        *,
        source: str,
        target: str,
        pathway_cfg: NeuralPathwayConfig,
    ) -> dict[str, Any]:
        payload = {
            "source": source,
            "target": target,
            "type": pathway_cfg.type,
            "strength": pathway_cfg.strength,
            "adaptation_rate": pathway_cfg.adaptation_rate,
            "bidirectional": pathway_cfg.type.lower() == "bidirectional",
        }
        await self._infra.eventbus.publish_raw(
            event_type="neural_pathway",
            source="consciousness_orchestrator",
            payload=payload,
        )
        return payload

    async def establish_constitutional_loops(self) -> list[dict[str, Any]]:
        cfg = self._config.infrastructure.constitutional
        loops = [
            {
                "name": "compliance_guard",
                "threshold": cfg.compliance_threshold,
                "auto_remediation": cfg.auto_remediation,
            },
            {
                "name": "evolution_monitor",
                "enabled": cfg.evolution_enabled,
            },
        ]
        await self._infra.eventbus.publish_raw(
            event_type="constitutional_feedback_loop",
            source="consciousness_orchestrator",
            payload={"loops": loops},
        )
        return loops

    async def initialize_reasoning_chains(self) -> list[dict[str, Any]]:
        chains = [
            {
                "name": "predictive_architecture",
                "sources": ["codex", "super_alita"],
                "targets": ["cma"],
            },
            {
                "name": "compliance_review",
                "sources": ["cma"],
                "targets": ["codex", "super_alita"],
            },
        ]
        await self._infra.eventbus.publish_raw(
            event_type="reasoning_chain",
            source="consciousness_orchestrator",
            payload={"chains": chains},
        )
        return chains

    async def enable_safe_self_modification(self) -> dict[str, Any]:
        cfg = self._config.consciousness.self_modification
        settings = {
            "enabled": cfg.enabled,
            "safety_checks": list(cfg.safety_checks),
            "approval_threshold": cfg.modification_approval_threshold,
        }
        await self._infra.eventbus.publish_raw(
            event_type="self_modification",
            source="consciousness_orchestrator",
            payload=settings,
        )
        return settings

    def _derive_endpoints(self, name: str) -> tuple[str, str]:
        if "_" not in name:
            return name, "consciousness"
        parts = name.split("_", 1)
        return parts[0], parts[1]


class ConsciousnessEmergenceFailure(RuntimeError):
    """Raised when the consciousness emergence score is below threshold."""


class UnifiedConsciousness:
    """Unified consciousness runtime with asynchronous coordination loops."""

    def __init__(
        self,
        *,
        infrastructure: SystemInfrastructure,
        components: dict[str, ComponentProcess],
        pathways: dict[str, dict[str, Any]],
        feedback_loops: list[dict[str, Any]],
        reasoning_chains: list[dict[str, Any]],
        self_modification: dict[str, Any],
        extra_loop_factories: Iterable[BackgroundFactory] | None = None,
    ) -> None:
        self.infrastructure = infrastructure
        self.components = components
        self.pathways = pathways
        self.feedback_loops = feedback_loops
        self.reasoning_chains = reasoning_chains
        self.self_modification_config = self_modification
        self._extra_loop_factories: list[BackgroundFactory] = list(
            extra_loop_factories or []
        )
        self._shutdown = asyncio.Event()
        self._violation_queue: asyncio.Queue[UnifiedEvent] = asyncio.Queue()
        self._learning_events: list[dict[str, Any]] = []
        self._current_score: float = 0.0

    async def bind_eventbus(self) -> None:
        async def _handle_violation(event: UnifiedEvent) -> None:
            await self._violation_queue.put(event)

        await self.infrastructure.eventbus.subscribe(
            "constitutional_violation",
            _handle_violation,
        )

    def add_background_task_factories(
        self, factories: Iterable[BackgroundFactory]
    ) -> None:
        self._extra_loop_factories.extend(factories)

    def is_running(self) -> bool:
        return not self._shutdown.is_set()

    async def assess_consciousness_emergence(self) -> float:
        ready_components = sum(
            1 for proc in self.components.values() if proc.status == "ready"
        )
        total_components = max(len(self.components), 1)
        ready_ratio = ready_components / total_components

        if self.pathways:
            strengths = [
                float(p.get("strength", 0.0)) for p in self.pathways.values()
            ]
            avg_strength = statistics.fmean(strengths)
        else:
            avg_strength = 0.5

        constitutional_threshold = (
            self.infrastructure.config.infrastructure.constitutional.compliance_threshold
        )

        score = statistics.fmean(
            [ready_ratio, avg_strength, constitutional_threshold]
        )
        self._current_score = score
        return score

    async def achieve_operational_coherence(self) -> None:
        observer = self.infrastructure.observability
        observer.log_run_started(
            {
                "run_id": "unified-consciousness",
                "session_id": "consciousness",
                "prompt": "bootstrap",
                "enable_planning": True,
            }
        )
        await self.infrastructure.eventbus.publish_raw(
            event_type="system_ready",
            source="unified_consciousness",
            payload={
                "components": list(self.components.keys()),
                "pathways": list(self.pathways.keys()),
                "score": self._current_score,
            },
        )

    async def run_forever(self) -> None:
        loops = [
            asyncio.create_task(
                self.constitutional_monitoring_loop(),
                name="constitutional_monitoring",
            ),
            asyncio.create_task(
                self.cross_component_learning_loop(),
                name="cross_component_learning",
            ),
            asyncio.create_task(
                self.predictive_reasoning_loop(), name="predictive_reasoning"
            ),
            asyncio.create_task(
                self.self_modification_loop(), name="self_modification"
            ),
            asyncio.create_task(
                self.human_interaction_loop(), name="human_interaction"
            ),
        ]

        for factory in list(self._extra_loop_factories):
            try:
                coroutine = factory()
            except Exception:
                continue
            loops.append(
                asyncio.create_task(
                    coroutine,
                    name=getattr(factory, "__name__", "advanced_capability"),
                )
            )

        try:
            await asyncio.gather(*loops)
        finally:
            for task in loops:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def constitutional_monitoring_loop(self) -> None:
        interval = 1.0
        threshold = (
            self.infrastructure.config.infrastructure.constitutional.compliance_threshold
        )
        while not self._shutdown.is_set():
            try:
                violation = await asyncio.wait_for(
                    self._violation_queue.get(), timeout=interval
                )
            except TimeoutError:
                continue

            score = (
                violation.payload.get("score", 1.0)
                if isinstance(violation.payload, dict)
                else 1.0
            )
            if score < threshold:
                remediation = await self.generate_remediation(
                    {
                        "score": score,
                        "event": violation.event_type,
                    }
                )
                await self.execute_auto_remediation(remediation)

    async def cross_component_learning_loop(self) -> None:
        interval = max(
            0.5,
            min(
                5.0,
                self.infrastructure.config.consciousness.learning.pattern_extraction_interval
                / 60,
            ),
        )
        while not self._shutdown.is_set():
            await asyncio.sleep(interval)
            opportunity = {
                "source_component": "codex",
                "target_components": ["super_alita", "cma"],
            }
            patterns = await self.extract_patterns(
                opportunity["source_component"]
            )
            for target in opportunity["target_components"]:
                enhanced = await self.transfer_learning(patterns, target)
                await self.integrate_enhanced_capability(target, enhanced)

    async def predictive_reasoning_loop(self) -> None:
        interval = 1.0
        while not self._shutdown.is_set():
            state = await self.assess_system_state()
            predictions = await self.predict_architectural_issues(state)
            for prediction in predictions:
                if (
                    prediction["severity"] > 0.8
                    and prediction["confidence"] > 0.7
                ):
                    action = await self.generate_preventive_action(prediction)
                    await self.execute_preventive_action(action)
            await asyncio.sleep(interval)

    async def self_modification_loop(self) -> None:
        if not self.self_modification_config.get("enabled", True):
            return
        interval = 2.0
        while not self._shutdown.is_set():
            await asyncio.sleep(interval)
            proposal = {
                "description": "refresh pathway strengths",
                "impact": 0.05,
            }
            if await self.evaluate_self_modification(proposal):
                await self.apply_self_modification(proposal)

    async def human_interaction_loop(self) -> None:
        interval = 2.0
        while not self._shutdown.is_set():
            await asyncio.sleep(interval)
            # Periodically emit status for operators.
            await self.infrastructure.eventbus.publish_raw(
                event_type="consciousness_metric",
                source="unified_consciousness",
                payload={
                    "score": self._current_score,
                    "timestamp": time.time(),
                },
            )

    async def generate_remediation(
        self, compliance: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "action": "increase_constitutional_monitoring",
            "details": compliance,
        }

    async def execute_auto_remediation(
        self, remediation: dict[str, Any]
    ) -> None:
        await self.infrastructure.eventbus.publish_raw(
            event_type="auto_remediation",
            source="unified_consciousness",
            payload=remediation,
        )

    async def detect_learning_opportunities(self) -> Iterable[dict[str, Any]]:
        return self._learning_events

    async def extract_patterns(self, component: str) -> dict[str, Any]:
        return {"component": component, "patterns": ["syntax", "workflow"]}

    async def transfer_learning(
        self, patterns: dict[str, Any], target_component: str
    ) -> dict[str, Any]:
        return {"target": target_component, "patterns": patterns["patterns"]}

    async def integrate_enhanced_capability(
        self, target: str, capability: dict[str, Any]
    ) -> None:
        await self.infrastructure.memory.put(
            target,
            {
                "timestamp": time.time(),
                "capability": capability,
            },
        )

    async def assess_system_state(self) -> dict[str, Any]:
        ready = sum(
            1 for proc in self.components.values() if proc.status == "ready"
        )
        degraded = sum(
            1 for proc in self.components.values() if proc.status == "failed"
        )
        return {
            "ready": ready,
            "degraded": degraded,
            "pathways": len(self.pathways),
            "score": self._current_score,
        }

    async def predict_architectural_issues(
        self, state: dict[str, Any]
    ) -> list[dict[str, float]]:
        if state["score"] < 0.6:
            return [
                {
                    "issue": "low_score",
                    "severity": 0.85,
                    "confidence": 0.75,
                }
            ]
        return []

    async def generate_preventive_action(
        self, prediction: dict[str, float]
    ) -> dict[str, Any]:
        return {
            "action": "increase_pathway_strength",
            "prediction": prediction,
        }

    async def execute_preventive_action(self, action: dict[str, Any]) -> None:
        await self.infrastructure.eventbus.publish_raw(
            event_type="preventive_action",
            source="unified_consciousness",
            payload=action,
        )

    async def evaluate_self_modification(
        self, proposal: dict[str, Any]
    ) -> bool:
        threshold = self.self_modification_config.get(
            "approval_threshold", 0.9
        )
        return proposal.get("impact", 0.0) <= threshold

    async def apply_self_modification(self, proposal: dict[str, Any]) -> None:
        await self.infrastructure.eventbus.publish_raw(
            event_type="self_modification_applied",
            source="unified_consciousness",
            payload=proposal,
        )

    def request_shutdown(self) -> None:
        self._shutdown.set()

    @property
    def current_score(self) -> float:
        return self._current_score


class ConsciousnessDashboard:
    """Render status snapshots for operators."""

    consciousness: UnifiedConsciousness

    async def render_status(self) -> dict[str, Any]:
        memory_report = await self.consciousness.infrastructure.memory.health()
        return {
            "consciousness_score": self.consciousness.current_score,
            "component_health": {
                name: proc.status
                for name, proc in self.consciousness.components.items()
            },
            "constitutional_compliance": self.consciousness.infrastructure.config.infrastructure.constitutional.compliance_threshold,
            "neural_pathway_strength": {
                name: cfg.get("strength", 0.0)
                for name, cfg in self.consciousness.pathways.items()
            },
            "active_predictions": 0,
            "learning_events_today": len(self.consciousness._learning_events),
            "self_modifications_applied": 0,
            "memory": memory_report,
        }


class EmergencyProtocols:
    """Respond to consciousness degradation scenarios."""

    def __init__(self, consciousness: UnifiedConsciousness) -> None:
        self._consciousness = consciousness

    async def handle_consciousness_degradation(self, score: float) -> None:
        if score < 0.3:
            await self.emergency_shutdown_and_restart()
        elif score < 0.5:
            await self.isolate_failing_components()
            await self.attempt_consciousness_recovery()
        elif score < 0.7:
            await self.increase_constitutional_monitoring()
            await self.boost_cross_component_communication()

    async def emergency_shutdown_and_restart(self) -> None:
        self._consciousness.request_shutdown()
        await self._consciousness.infrastructure.eventbus.publish_raw(
            event_type="emergency_shutdown",
            source="emergency_protocols",
            payload={"reason": "critical_score"},
        )

    async def isolate_failing_components(self) -> None:
        failed = [
            name
            for name, proc in self._consciousness.components.items()
            if proc.status == "failed"
        ]
        await self._consciousness.infrastructure.eventbus.publish_raw(
            event_type="isolate_components",
            source="emergency_protocols",
            payload={"components": failed},
        )

    async def attempt_consciousness_recovery(self) -> None:
        await self._consciousness.infrastructure.eventbus.publish_raw(
            event_type="attempt_recovery",
            source="emergency_protocols",
            payload={"score": self._consciousness.current_score},
        )

    async def increase_constitutional_monitoring(self) -> None:
        await self._consciousness.infrastructure.eventbus.publish_raw(
            event_type="increase_monitoring",
            source="emergency_protocols",
            payload={"metric": "constitutional"},
        )

    async def boost_cross_component_communication(self) -> None:
        await self._consciousness.infrastructure.eventbus.publish_raw(
            event_type="boost_communication",
            source="emergency_protocols",
            payload={
                "components": list(self._consciousness.components.keys())
            },
        )
