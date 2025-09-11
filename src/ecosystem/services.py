"""
Decoupled services that subscribe to ecosystem events to provide
observability and metrics recording.
"""

from __future__ import annotations

from .master_orchestrator import (
    IEventBus,
    IMetricsCollector,
    WorkflowCompletedEvent,
)


class ObservabilityService:
    """
    Observability service that listens to workflow events and records metrics.
    Fully decoupled from the orchestrator via the event bus.
    """

    def __init__(self, event_bus: IEventBus, metrics_collector: IMetricsCollector):
        self.metrics_collector = metrics_collector
        event_bus.subscribe(WorkflowCompletedEvent, self.handle_workflow_completion)

    async def handle_workflow_completion(self, event: WorkflowCompletedEvent) -> None:
        """Handler for WorkflowCompletedEvent events."""
        await self.metrics_collector.record_workflow_execution(
            workflow_name=event.workflow_name,
            metadata=event.metrics,
        )

