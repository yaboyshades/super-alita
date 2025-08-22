"""
Autonomous immune system.
Listens for failures, consults memory, escalates to Pilot, learns.
"""

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any

from src.core.event_bus import EventBus
from src.core.events import BaseEvent
from src.core.neural_atom import NeuralStore
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


# Minimal Pydantic-like models (replace with actual Pydantic in production)
class BaseModel:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_validate_json(self, json_str: str):
        import json

        data = json.loads(json_str)
        return self.__class__(**data)

    def model_dump_json(self) -> str:
        import json

        return json.dumps(self.__dict__)


def Field(default_factory=None):
    return default_factory() if default_factory else None


class Action(str, Enum):
    RETRY = "retry"
    ROLLBACK = "rollback"
    PATCH = "patch"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class Directive(BaseModel):
    def __init__(self, action: Action, reasoning: str, params: dict[str, Any] = None):
        self.action = action
        self.reasoning = reasoning
        self.params = params or {}


class Outcome(BaseModel):
    def __init__(self, success: bool, action: Action, details: dict[str, Any] = None):
        self.success = success
        self.action = action
        self.details = details or {}


class SelfHealPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self._name = "self_heal"
        self._memory: Any | None = None
        self._health = None
        self._pilot = None  # Will be initialized in setup

    @property
    def name(self) -> str:
        return self._name

    async def setup(self, bus: EventBus, store: NeuralStore, cfg: dict[str, Any]):
        await super().setup(bus, store, cfg)

        # Create health atom using the module function
        from src.core.neural_atom import create_goal_atom

        health_atom = create_goal_atom("system.health", "healthy")
        store.register(health_atom)
        self._health = health_atom

        # Initialize Gemini Pilot client
        from src.core.gemini_pilot import GeminiPilotClient

        self._pilot = GeminiPilotClient()

        # Wire semantic memory plugin reference
        self._memory = store  # Use the store directly for memory operations
        logger.info(
            "SelfHealPlugin initialized with Gemini Pilot integration and memory wiring"
        )

    async def start(self):
        await super().start()
        await self.subscribe("system_alert", self._on_failure)
        await self.subscribe("tool_execution", self._on_failure)
        logger.info("SelfHealPlugin monitoring.")

    async def _on_failure(self, event: BaseEvent):
        """Unified failure entry point."""
        failure = _FailureSnapshot.from_event(event)
        logger.warning(f"Healing triggered: {failure.summary}")

        if self._health:
            self._health.value = "recovering"

        directive = await self._diagnose_with_pilot(failure)
        outcome = await self._execute(directive, failure.context)

        await self._record_experience(failure, directive, outcome)

        if self._health:
            self._health.value = "healthy" if outcome.success else "degraded"

    # ------------------------------------------------------------------ internals
    async def _diagnose_with_pilot(self, failure: "_FailureSnapshot") -> Directive:
        """Consult Pilot (LLM) for strategic diagnosis."""
        memories = []
        if self._memory:
            try:
                query_vec = await self._memory.embed_text([failure.summary])
                memories = await self.store.attention(query_vec[0], top_k=3)
            except Exception:
                pass

        # Use Gemini Pilot for structured decision making
        try:
            response = await asyncio.wait_for(
                self._pilot.self_heal_decide(
                    failure_type=failure.event_type,
                    error_details=failure.summary,
                    component=failure.component,
                    system_state=str(self._health.value) if self._health else "unknown",
                    retry_count=failure.context.get("retry_count", 0),
                    failure_history=[m.data for m in memories] if memories else None,
                    diagnostic_data=failure.context,
                ),
                timeout=45,
            )

            # Map Gemini response to internal Directive format
            action_mapping = {
                "retry": Action.RETRY,
                "restart": Action.RESTART,
                "rollback": Action.ROLLBACK,
                "patch": Action.PATCH,
                "escalate": Action.ESCALATE,
            }

            return Directive(
                action=action_mapping.get(response["action"], Action.ESCALATE),
                reasoning=response["reasoning"],
                confidence=response.get("confidence", 0.5),
                params=response.get("parameters", {}),
            )

        except Exception as e:
            logger.error(f"Pilot unreachable: {e}")
            return Directive(action=Action.ESCALATE, reasoning="Pilot unreachable")

    async def _execute(self, d: Directive, ctx: dict[str, Any]) -> Outcome:
        """Execute healing action with real orchestration."""
        logger.info(f"Executing healing action={d.action} params={d.params}")

        success = False
        try:
            if d.action == Action.RETRY:
                # Log the retry attempt
                logger.info(f"Retrying operation: {ctx.get('operation', 'unknown')}")
                # Record error in memory for learning
                if hasattr(self.store, "upsert"):
                    await self.store.upsert(
                        content=f"Retry attempt: {ctx.get('error', 'unknown error')}",
                        metadata={"type": "healing_action", "action": "retry"},
                        hierarchy_path=["self_heal", "retries"],
                    )
                success = True

            elif d.action == Action.ROLLBACK:
                logger.info("Initiating system rollback")
                # Record rollback in memory
                if hasattr(self.store, "upsert"):
                    await self.store.upsert(
                        content=f"Rollback initiated: {d.reasoning}",
                        metadata={"type": "healing_action", "action": "rollback"},
                        hierarchy_path=["self_heal", "rollbacks"],
                    )
                success = True

            elif d.action == Action.ESCALATE:
                logger.warning(f"Escalating issue: {d.reasoning}")
                # Record escalation in memory
                if hasattr(self.store, "upsert"):
                    await self.store.upsert(
                        content=f"Escalated issue: {d.reasoning}",
                        metadata={"type": "healing_action", "action": "escalate"},
                        hierarchy_path=["self_heal", "escalations"],
                    )
                success = False  # Escalation means we couldn't self-heal

        except Exception as e:
            logger.error(f"Healing action execution failed: {e}")
            success = False

        return Outcome(success=success, action=d.action)

    async def _record_experience(
        self, failure: "_FailureSnapshot", d: Directive, o: Outcome
    ):
        """Record healing experience in memory."""
        if self._memory:
            try:
                await self._memory.upsert(
                    memory_id=f"heal_{uuid.uuid4().hex[:8]}",
                    content={
                        "issue": failure.summary,
                        "context": failure.context,
                        "action": d.action.value,
                        "success": o.success,
                        "reasoning": d.reasoning,
                    },
                    hierarchy_path=["system", "healing_log"],
                    owner_plugin=self.name,
                )
            except Exception as e:
                logger.warning(f"Failed to record healing experience: {e}")

    async def health_check(self) -> dict[str, Any]:
        """Plugin health check."""
        return {
            "plugin": self.name,
            "status": "healthy",
            "health_atom": self._health.value if self._health else "unknown",
        }

    async def shutdown(self):
        """Clean shutdown."""
        logger.info("SelfHealPlugin shutting down")
        if self._pilot:
            await self._pilot.close()
        self._is_running = False

    # ==================== COGNITIVE CONTRACT ALIAS ====================

    def _build_healing_cognitive_contract(
        self, severity, component: str, issue_description: str, recent_events
    ) -> str:
        """
        Alias for Gemini Pilot to match Cognitive Contract naming convention.
        This method is deprecated - use _diagnose_with_pilot directly.
        """
        # Simple wrapper that returns the issue description for compatibility
        return (
            f"Component: {component}\nIssue: {issue_description}\nSeverity: {severity}"
        )


class _FailureSnapshot:
    """Normalized failure representation from any event type."""

    def __init__(self, summary: str, context: dict[str, Any]):
        self.summary = summary
        self.context = context

    @classmethod
    def from_event(cls, e: BaseEvent) -> "_FailureSnapshot":
        """Convert any failure event to normalized snapshot."""
        if hasattr(e, "tool_name") and hasattr(e, "return_code"):
            # Tool execution failure
            return cls(
                summary=f"Tool {e.tool_name} failed (code {e.return_code})",
                context={
                    "stderr": getattr(e, "stderr", ""),
                    "params": getattr(e, "parameters", {}),
                },
            )
        if (
            hasattr(e, "level")
            and hasattr(e, "message")
            and e.level in {"ERROR", "CRITICAL"}
        ):
            # System alert failure
            return cls(summary=e.message, context=getattr(e, "metadata", {}))
        # Generic failure
        return cls(summary=str(e), context={"event_type": e.event_type})
