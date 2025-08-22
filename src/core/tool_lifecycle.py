#!/usr/bin/env python3
"""
Tool Lifecycle Management (TLM) System
Implements comprehensive tool lifecycle management including creation, registration,
activation, monitoring, deactivation, and cleanup with robust state transitions.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# Note: These imports assume the existence of these modules elsewhere in the codebase.
# They are included here as per the provided text's implementation plan.
# from core.events import create_event
# from core.plugin_communication import (
#     MessagePriority,
#     PluginCommunicationHub,
#     PluginDependency,
#     PluginMessage,
# )
# from core.response_router import CapabilityMatch, ResponseRouter

# --- Enums ---


class ToolState(Enum):
    """Tool lifecycle states"""

    CREATED = "created"
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    DEGRADED = "degraded"
    FAILING = "failing"
    DEACTIVATING = "deactivating"
    INACTIVE = "inactive"
    DESTROYED = "destroyed"


class HealthStatus(Enum):
    """Tool health status levels"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    UNREACHABLE = "unreachable"


class DependencyStatus(Enum):
    """Dependency resolution status"""

    SATISFIED = "satisfied"
    PARTIAL = "partial"
    UNSATISFIED = "unsatisfied"
    CIRCULAR = "circular"
    FAILED = "failed"


# --- Data Classes ---


@dataclass
class ToolMetrics:
    """Tool performance and health metrics"""

    tool_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    success_rate: float = 100.0
    error_count: int = 0
    request_count: int = 0
    last_activity: datetime = field(default_factory=lambda: datetime.now(UTC))
    uptime_seconds: float = 0.0
    restart_count: int = 0

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now(UTC)

    def record_request(self, success: bool, response_time: float) -> None:
        """Record a request execution"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        # Update success rate
        self.success_rate = (
            (self.request_count - self.error_count) / self.request_count * 100
            if self.request_count > 0
            else 100.0
        )
        # Update response time (simple moving average)
        alpha = 0.1  # Smoothing factor
        if self.response_time_avg == 0.0:
            self.response_time_avg = response_time
        else:
            self.response_time_avg = (
                alpha * response_time + (1 - alpha) * self.response_time_avg
            )
        self.update_activity()


@dataclass
class ToolDefinition:
    """Definition and configuration for a tool"""

    tool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    category: str = "general"
    # Capabilities and requirements
    provides_capabilities: list[str] = field(default_factory=list)
    requires_capabilities: list[str] = field(default_factory=list)
    optional_capabilities: list[str] = field(default_factory=list)
    # Resource requirements
    min_memory_mb: int = 128
    max_memory_mb: int = 1024
    min_cpu_cores: float = 0.1
    max_cpu_cores: float = 2.0
    disk_space_mb: int = 100
    # Lifecycle configuration
    startup_timeout_seconds: int = 30
    shutdown_timeout_seconds: int = 10
    health_check_interval_seconds: int = 30
    max_restart_attempts: int = 3
    auto_restart: bool = True
    # Integration points
    plugin_class: str = ""
    config_schema: dict[str, Any] = field(default_factory=dict)
    initialization_params: dict[str, Any] = field(default_factory=dict)
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolInstance:
    """Runtime instance of a tool"""

    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_definition: ToolDefinition = None
    state: ToolState = ToolState.CREATED
    health_status: HealthStatus = HealthStatus.UNKNOWN
    # Runtime data
    process_id: str = ""
    plugin_instance: Any = None
    started_at: datetime | None = None
    last_health_check: datetime | None = None
    health_check_failures: int = 0
    # Dependencies
    dependency_status: DependencyStatus = DependencyStatus.UNSATISFIED
    resolved_dependencies: list[str] = field(default_factory=list)
    failed_dependencies: list[str] = field(default_factory=list)
    # Metrics and monitoring
    metrics: ToolMetrics = None
    error_log: list[dict[str, Any]] = field(default_factory=list)
    state_history: list[tuple[ToolState, datetime, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.metrics is None and self.tool_definition:
            self.metrics = ToolMetrics(tool_id=self.tool_definition.tool_id)

    def transition_state(self, new_state: ToolState, reason: str = "") -> bool:
        """Transition to a new state with validation"""
        valid_transitions = {
            ToolState.CREATED: [ToolState.REGISTERED, ToolState.DESTROYED],
            ToolState.REGISTERED: [
                ToolState.INITIALIZING,
                ToolState.INACTIVE,
                ToolState.DESTROYED,
            ],
            ToolState.INITIALIZING: [
                ToolState.ACTIVE,
                ToolState.FAILING,
                ToolState.INACTIVE,
            ],
            ToolState.ACTIVE: [
                ToolState.PAUSED,
                ToolState.DEGRADED,
                ToolState.FAILING,
                ToolState.DEACTIVATING,
            ],
            ToolState.PAUSED: [ToolState.ACTIVE, ToolState.DEACTIVATING],
            ToolState.DEGRADED: [
                ToolState.ACTIVE,
                ToolState.FAILING,
                ToolState.DEACTIVATING,
            ],
            ToolState.FAILING: [
                ToolState.ACTIVE,
                ToolState.INACTIVE,
                ToolState.DEACTIVATING,
            ],
            ToolState.DEACTIVATING: [ToolState.INACTIVE, ToolState.DESTROYED],
            ToolState.INACTIVE: [
                ToolState.INITIALIZING,
                ToolState.DESTROYED,
            ],
            ToolState.DESTROYED: [],  # Terminal state
        }
        if new_state in valid_transitions.get(self.state, []):
            # Record state change
            self.state_history.append((self.state, datetime.now(UTC), reason))
            self.state = new_state
            # Update timestamps
            if new_state == ToolState.ACTIVE and self.started_at is None:
                self.started_at = datetime.now(UTC)
            return True
        return False

    def record_error(self, error_type: str, message: str, details: dict = None) -> None:
        """Record an error for this tool instance"""
        error_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": error_type,
            "message": message,
            "details": details or {},
            "state": self.state.value,
        }
        self.error_log.append(error_entry)
        # Keep only recent errors (last 100)
        if len(self.error_log) > 100:
            self.error_log.pop(0)
        # Update metrics
        if self.metrics:
            self.metrics.error_count += 1

    def is_healthy(self) -> bool:
        """Check if tool instance is in a healthy state"""
        return (
            self.state == ToolState.ACTIVE
            and self.health_status == HealthStatus.HEALTHY
            and self.dependency_status == DependencyStatus.SATISFIED
        )

    def needs_restart(self) -> bool:
        """Check if tool needs to be restarted"""
        if not self.tool_definition or not self.tool_definition.auto_restart:
            return False
        # Check if in failing state
        if self.state == ToolState.FAILING:
            return (
                self.metrics.restart_count < self.tool_definition.max_restart_attempts
            )
        # Check if health checks are failing
        if self.health_check_failures >= 3:
            return (
                self.metrics.restart_count < self.tool_definition.max_restart_attempts
            )
        return False


# --- Core Components ---


class ToolRegistry:
    """Registry for tool definitions and instances"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Storage
        self.tool_definitions: dict[str, ToolDefinition] = {}
        self.tool_instances: dict[str, ToolInstance] = {}
        self.name_to_id_mapping: dict[str, str] = {}
        # Indexing for efficient lookups
        self.capability_index: dict[str, set[str]] = {}  # capability -> tool_ids
        self.category_index: dict[str, set[str]] = {}  # category -> tool_ids
        self.tag_index: dict[str, set[str]] = {}  # tag -> tool_ids

    def register_tool_definition(self, definition: ToolDefinition) -> bool:
        """Register a new tool definition"""
        if definition.tool_id in self.tool_definitions:
            self.logger.warning(f"Tool definition already exists: {definition.tool_id}")
            return False
        # Validate definition
        if not definition.name or not definition.description:
            self.logger.error("Tool definition missing required fields")
            return False
        # Store definition
        self.tool_definitions[definition.tool_id] = definition
        self.name_to_id_mapping[definition.name] = definition.tool_id
        # Update indexes
        self._update_indexes(definition)
        self.logger.info(f"Registered tool definition: {definition.name}")
        return True

    def create_tool_instance(self, tool_id: str) -> ToolInstance | None:
        """Create a new instance of a tool"""
        if tool_id not in self.tool_definitions:
            self.logger.error(f"Tool definition not found: {tool_id}")
            return None
        definition = self.tool_definitions[tool_id]
        instance = ToolInstance(tool_definition=definition)
        # Initialize state
        instance.transition_state(ToolState.REGISTERED, "Instance created")
        # Store instance
        self.tool_instances[instance.instance_id] = instance
        self.logger.info(f"Created tool instance: {instance.instance_id}")
        return instance

    def get_tool_by_name(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name"""
        tool_id = self.name_to_id_mapping.get(name)
        return self.tool_definitions.get(tool_id) if tool_id else None

    def get_tool_instance(self, instance_id: str) -> ToolInstance | None:
        """Get tool instance by ID"""
        return self.tool_instances.get(instance_id)

    def find_tools_by_capability(self, capability: str) -> list[ToolDefinition]:
        """Find tools that provide a specific capability"""
        tool_ids = self.capability_index.get(capability, set())
        return [self.tool_definitions[tool_id] for tool_id in tool_ids]

    def find_tools_by_category(self, category: str) -> list[ToolDefinition]:
        """Find tools in a specific category"""
        tool_ids = self.category_index.get(category, set())
        return [self.tool_definitions[tool_id] for tool_id in tool_ids]

    def find_tools_by_tag(self, tag: str) -> list[ToolDefinition]:
        """Find tools with a specific tag"""
        tool_ids = self.tag_index.get(tag, set())
        return [self.tool_definitions[tool_id] for tool_id in tool_ids]

    def get_active_instances(self) -> list[ToolInstance]:
        """Get all active tool instances"""
        return [
            instance
            for instance in self.tool_instances.values()
            if instance.state == ToolState.ACTIVE
        ]

    def get_instances_by_state(self, state: ToolState) -> list[ToolInstance]:
        """Get all instances in a specific state"""
        return [
            instance
            for instance in self.tool_instances.values()
            if instance.state == state
        ]

    def remove_tool_instance(self, instance_id: str) -> bool:
        """Remove a tool instance from registry"""
        if instance_id not in self.tool_instances:
            return False
        instance = self.tool_instances[instance_id]
        if instance.state not in [ToolState.INACTIVE, ToolState.DESTROYED]:
            self.logger.warning(
                f"Removing instance {instance_id} in state {instance.state}"
            )
        del self.tool_instances[instance_id]
        self.logger.info(f"Removed tool instance: {instance_id}")
        return True

    def _update_indexes(self, definition: ToolDefinition) -> None:
        """Update search indexes for a tool definition"""
        tool_id = definition.tool_id
        # Capability index
        for capability in definition.provides_capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = set()
            self.capability_index[capability].add(tool_id)
        # Category index
        if definition.category not in self.category_index:
            self.category_index[definition.category] = set()
        self.category_index[definition.category].add(tool_id)
        # Tag index
        for tag in definition.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(tool_id)

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        state_counts = {}
        for instance in self.tool_instances.values():
            state = instance.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        return {
            "total_definitions": len(self.tool_definitions),
            "total_instances": len(self.tool_instances),
            "state_distribution": state_counts,
            "categories": list(self.category_index.keys()),
            "capabilities": list(self.capability_index.keys()),
            "tags": list(self.tag_index.keys()),
        }


class ToolHealthMonitor:
    """Monitors tool health and performance"""

    def __init__(self, registry: ToolRegistry, event_bus):
        self.registry = registry
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        # Monitoring state
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._health_check_handlers: dict[str, Callable] = {}

    async def start(self) -> None:
        """Start health monitoring"""
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Tool health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring"""
        if not self._running:
            return
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Tool health monitor stopped")

    def register_health_check(
        self, tool_name: str, health_check_func: Callable
    ) -> None:
        """Register a custom health check function for a tool"""
        self._health_check_handlers[tool_name] = health_check_func
        self.logger.debug(f"Registered health check for tool: {tool_name}")

    async def check_tool_health(self, instance: ToolInstance) -> HealthStatus:
        """Check health of a specific tool instance"""
        if not instance.tool_definition:
            return HealthStatus.UNKNOWN
        tool_name = instance.tool_definition.name
        try:
            # Use custom health check if available
            if tool_name in self._health_check_handlers:
                handler = self._health_check_handlers[tool_name]
                if asyncio.iscoroutinefunction(handler):
                    health_result = await handler(instance)
                else:
                    health_result = handler(instance)
                if isinstance(health_result, HealthStatus):
                    return health_result
                elif isinstance(health_result, bool):
                    return (
                        HealthStatus.HEALTHY if health_result else HealthStatus.CRITICAL
                    )
                elif isinstance(health_result, dict):
                    return self._parse_health_result(health_result)
            # Default health checks
            return await self._default_health_check(instance)
        except Exception as e:
            self.logger.error(f"Health check failed for {tool_name}: {e}")
            instance.record_error("health_check_error", str(e))
            return HealthStatus.CRITICAL

    async def _default_health_check(self, instance: ToolInstance) -> HealthStatus:
        """Default health check implementation"""
        # Check if instance is in valid state
        if instance.state not in [ToolState.ACTIVE, ToolState.DEGRADED]:
            return HealthStatus.CRITICAL
        # Check last activity
        if instance.metrics and instance.metrics.last_activity:
            time_since_activity = (
                datetime.now(UTC) - instance.metrics.last_activity
            ).total_seconds()
            # If no activity for more than 5 minutes, warn
            if time_since_activity > 300:
                return HealthStatus.WARNING
        # Check error rate
        if instance.metrics and instance.metrics.success_rate < 90:
            return HealthStatus.WARNING
        elif instance.metrics and instance.metrics.success_rate < 50:
            return HealthStatus.CRITICAL
        # Check response time
        if instance.metrics and instance.metrics.response_time_avg > 10.0:
            return HealthStatus.WARNING
        return HealthStatus.HEALTHY

    def _parse_health_result(self, result: dict[str, Any]) -> HealthStatus:
        """Parse health check result dictionary"""
        status = result.get("status", "unknown").lower()
        status_mapping = {
            "healthy": HealthStatus.HEALTHY,
            "warning": HealthStatus.WARNING,
            "critical": HealthStatus.CRITICAL,
            "unreachable": HealthStatus.UNREACHABLE,
        }
        return status_mapping.get(status, HealthStatus.UNKNOWN)

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self._running:
            try:
                # Check all active instances
                active_instances = self.registry.get_active_instances()
                for instance in active_instances:
                    try:
                        # Perform health check
                        new_health_status = await self.check_tool_health(instance)
                        # Update instance health
                        old_health_status = instance.health_status
                        instance.health_status = new_health_status
                        instance.last_health_check = datetime.now(UTC)
                        # Handle health status changes
                        if old_health_status != new_health_status:
                            await self._handle_health_status_change(
                                instance, old_health_status, new_health_status
                            )
                        # Update health check failure count
                        if new_health_status in [
                            HealthStatus.CRITICAL,
                            HealthStatus.UNREACHABLE,
                        ]:
                            instance.health_check_failures += 1
                        else:
                            instance.health_check_failures = 0
                        # Check if restart is needed
                        if instance.needs_restart():
                            await self._trigger_restart(instance)
                    except Exception as e:
                        self.logger.error(
                            f"Error monitoring instance {instance.instance_id}: {e}"
                        )
                # Wait before next check cycle
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Longer wait on error

    async def _handle_health_status_change(
        self,
        instance: ToolInstance,
        old_status: HealthStatus,
        new_status: HealthStatus,
    ) -> None:
        """Handle health status changes"""
        self.logger.info(
            f"Tool {instance.tool_definition.name} health changed: {old_status.value} -> {new_status.value}"
        )
        # Emit health change event
        try:
            # Assuming create_event exists and event_bus.emit works
            # event = create_event(
            #     "tool_health_changed",
            #     instance_id=instance.instance_id,
            #     tool_name=instance.tool_definition.name,
            #     old_status=old_status.value,
            #     new_status=new_status.value,
            #     timestamp=datetime.now(UTC).isoformat(),
            #     source_plugin="tool_health_monitor",
            # )
            # await self.event_bus.emit(event)
            pass  # Placeholder for actual event emission
        except Exception as e:
            self.logger.warning(f"Could not emit health change event: {e}")
        # Update tool state based on health
        if new_status == HealthStatus.CRITICAL:
            instance.transition_state(ToolState.FAILING, "Health check critical")
        elif new_status == HealthStatus.WARNING and instance.state == ToolState.ACTIVE:
            instance.transition_state(ToolState.DEGRADED, "Health check warning")
        elif (
            new_status == HealthStatus.HEALTHY and instance.state == ToolState.DEGRADED
        ):
            instance.transition_state(ToolState.ACTIVE, "Health recovered")

    async def _trigger_restart(self, instance: ToolInstance) -> None:
        """Trigger restart for a failing tool instance"""
        self.logger.info(
            f"Triggering restart for tool: {instance.tool_definition.name}"
        )
        try:
            # Emit restart event
            # event = create_event(
            #     "tool_restart_triggered",
            #     instance_id=instance.instance_id,
            #     tool_name=instance.tool_definition.name,
            #     restart_count=instance.metrics.restart_count,
            #     reason="Health check failure",
            #     source_plugin="tool_health_monitor",
            # )
            # await self.event_bus.emit(event)
            # Update restart count
            instance.metrics.restart_count += 1
            pass  # Placeholder for actual event emission and restart logic
        except Exception as e:
            self.logger.warning(f"Could not emit restart event: {e}")


class DependencyResolver:
    """Resolves tool dependencies"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)

    def resolve_dependencies(self, tool_instance: ToolInstance) -> DependencyStatus:
        """Resolve dependencies for a tool instance"""
        if not tool_instance.tool_definition:
            return DependencyStatus.FAILED
        required_caps = tool_instance.tool_definition.requires_capabilities
        optional_caps = tool_instance.tool_definition.optional_capabilities
        if not required_caps and not optional_caps:
            tool_instance.dependency_status = DependencyStatus.SATISFIED
            return DependencyStatus.SATISFIED
        # Get available capabilities from active tools
        available_capabilities = self._get_available_capabilities()
        # Check required dependencies
        satisfied_required = set(required_caps).issubset(available_capabilities)
        satisfied_optional = set(optional_caps).intersection(available_capabilities)
        # Update instance dependency info
        tool_instance.resolved_dependencies = list(
            set(required_caps).intersection(available_capabilities)
        ) + list(satisfied_optional)
        tool_instance.failed_dependencies = list(
            set(required_caps) - available_capabilities
        )
        # Determine status
        if satisfied_required:
            if len(satisfied_optional) > 0:
                status = DependencyStatus.SATISFIED
            else:
                status = DependencyStatus.SATISFIED
        elif len(tool_instance.resolved_dependencies) > 0:
            status = DependencyStatus.PARTIAL
        else:
            status = DependencyStatus.UNSATISFIED
        tool_instance.dependency_status = status
        return status

    def check_circular_dependencies(
        self, tool_definitions: list[ToolDefinition]
    ) -> list[list[str]]:
        """Check for circular dependencies among tools"""
        # Build dependency graph
        graph = {}
        for tool_def in tool_definitions:
            graph[tool_def.tool_id] = set()
            # Find tools that provide required capabilities
            for required_cap in tool_def.requires_capabilities:
                for other_tool in tool_definitions:
                    if (
                        required_cap in other_tool.provides_capabilities
                        and other_tool.tool_id != tool_def.tool_id
                    ):
                        graph[tool_def.tool_id].add(other_tool.tool_id)
        # Detect cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node_id: str, path: list[str]) -> None:
            if node_id in rec_stack:
                # Found cycle
                cycle_start = path.index(node_id)
                cycle = path[cycle_start:] + [node_id]
                cycles.append(cycle)
                return
            if node_id in visited:
                return
            visited.add(node_id)
            rec_stack.add(node_id)
            for neighbor in graph.get(node_id, set()):
                dfs(neighbor, path + [node_id])
            rec_stack.remove(node_id)

        for tool_id in graph:
            if tool_id not in visited:
                dfs(tool_id, [])
        return cycles

    def get_dependency_order(
        self, tool_definitions: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """Get tools ordered by dependencies (topological sort)"""
        # Build dependency graph
        graph = {}
        in_degree = {}
        for tool_def in tool_definitions:
            graph[tool_def.tool_id] = []
            in_degree[tool_def.tool_id] = 0
        # Add edges
        for tool_def in tool_definitions:
            for required_cap in tool_def.requires_capabilities:
                for provider_tool in tool_definitions:
                    if (
                        required_cap in provider_tool.provides_capabilities
                        and provider_tool.tool_id != tool_def.tool_id
                    ):
                        graph[provider_tool.tool_id].append(tool_def.tool_id)
                        in_degree[tool_def.tool_id] += 1
        # Topological sort using Kahn's algorithm
        queue = [tool_id for tool_id, degree in in_degree.items() if degree == 0]
        result = []
        while queue:
            current = queue.pop(0)
            result.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        # Convert IDs back to definitions
        id_to_def = {tool_def.tool_id: tool_def for tool_def in tool_definitions}
        return [id_to_def[tool_id] for tool_id in result if tool_id in id_to_def]

    def _get_available_capabilities(self) -> set[str]:
        """Get all capabilities available from active tools"""
        capabilities = set()
        for instance in self.registry.get_active_instances():
            if (
                instance.tool_definition
                and instance.health_status == HealthStatus.HEALTHY
            ):
                capabilities.update(instance.tool_definition.provides_capabilities)
        return capabilities


class IntelligentToolSelector:
    """Intelligent tool selection based on context and performance"""

    def __init__(
        self, registry: ToolRegistry, response_router
    ):  # Assuming response_router exists
        self.registry = registry
        self.response_router = response_router
        self.logger = logging.getLogger(__name__)
        # Selection strategies
        self.selection_history: dict[str, list[dict[str, Any]]] = {}

    def select_tools_for_request(
        self, user_input: str, conversation_id: str, context: dict[str, Any] = None
    ) -> list[ToolInstance]:
        """Select optimal tools for a user request"""
        context = context or {}
        # Use response router to get capability matches
        routing_decision = None
        try:
            # This would be async in real implementation
            # routing_decision = await self.response_router.route_request(user_input, conversation_id)
            pass  # Placeholder for actual routing decision
        except Exception as e:
            self.logger.warning(f"Could not get routing decision: {e}")
        # Find tools that match required capabilities
        selected_tools = []
        # Extract keywords from user input for capability matching
        keywords = self._extract_keywords(user_input)
        relevant_capabilities = self._map_keywords_to_capabilities(keywords)
        for capability in relevant_capabilities:
            tool_definitions = self.registry.find_tools_by_capability(capability)
            for tool_def in tool_definitions:
                # Find active instances of this tool
                active_instances = [
                    instance
                    for instance in self.registry.get_active_instances()
                    if (
                        instance.tool_definition
                        and instance.tool_definition.tool_id == tool_def.tool_id
                        and instance.is_healthy()
                    )
                ]
                if active_instances:
                    # Select best instance based on performance
                    best_instance = self._select_best_instance(active_instances)
                    if best_instance not in selected_tools:
                        selected_tools.append(best_instance)
        # If no tools found, try fallback selection
        if not selected_tools:
            selected_tools = self._fallback_selection(user_input, context)
        # Record selection for learning
        self._record_selection(conversation_id, user_input, selected_tools)
        return selected_tools

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from user input"""
        # Simple keyword extraction (could be enhanced with NLP)
        import re

        text_lower = text.lower()
        keywords = re.findall(r"\b\w+\b", text_lower)
        # Filter out common words
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "this",
            "that",
            "these",
            "those",
        }
        return [word for word in keywords if word not in stopwords and len(word) > 2]

    def _map_keywords_to_capabilities(self, keywords: list[str]) -> set[str]:
        """Map keywords to tool capabilities"""
        keyword_capability_map = {
            "create": ["creation", "generation"],
            "make": ["creation", "generation"],
            "build": ["creation", "generation"],
            "generate": ["creation", "generation"],
            "analyze": ["analysis", "inspection"],
            "examine": ["analysis", "inspection"],
            "review": ["analysis", "inspection"],
            "check": ["analysis", "validation"],
            "debug": ["debugging", "error_handling"],
            "fix": ["debugging", "repair"],
            "error": ["debugging", "error_handling"],
            "problem": ["debugging", "problem_solving"],
            "search": ["search", "discovery"],
            "find": ["search", "discovery"],
            "plan": ["planning", "organization"],
            "organize": ["planning", "organization"],
            "refactor": ["refactoring", "optimization"],
            "optimize": ["optimization", "performance"],
            "monitor": ["monitoring", "metrics"],
            "measure": ["metrics", "measurement"],
        }
        capabilities = set()
        for keyword in keywords:
            if keyword in keyword_capability_map:
                capabilities.update(keyword_capability_map[keyword])
        return capabilities

    def _select_best_instance(self, instances: list[ToolInstance]) -> ToolInstance:
        """Select the best instance based on performance metrics"""
        if len(instances) == 1:
            return instances[0]
        # Score instances based on metrics
        scored_instances = []
        for instance in instances:
            score = 0.0
            if instance.metrics:
                # Higher success rate is better
                score += instance.metrics.success_rate / 100.0 * 0.4
                # Lower response time is better
                if instance.metrics.response_time_avg > 0:
                    score += (1.0 / instance.metrics.response_time_avg) * 0.3
                # Recent activity is good
                if instance.metrics.last_activity:
                    hours_since_activity = (
                        datetime.now(UTC) - instance.metrics.last_activity
                    ).total_seconds() / 3600
                    activity_score = max(0, 1.0 - (hours_since_activity / 24))
                    score += activity_score * 0.2
                # Lower restart count is better
                restart_penalty = min(instance.metrics.restart_count * 0.1, 0.1)
                score -= restart_penalty
            scored_instances.append((score, instance))
        # Sort by score (highest first)
        scored_instances.sort(key=lambda x: x[0], reverse=True)
        return scored_instances[0][1]

    def _fallback_selection(
        self, user_input: str, context: dict[str, Any]
    ) -> list[ToolInstance]:
        """Fallback tool selection when no specific matches found"""
        # Try to find general-purpose tools
        general_tools = self.registry.find_tools_by_category("general")
        active_general = []
        for tool_def in general_tools:
            active_instances = [
                instance
                for instance in self.registry.get_active_instances()
                if (
                    instance.tool_definition
                    and instance.tool_definition.tool_id == tool_def.tool_id
                    and instance.is_healthy()
                )
            ]
            if active_instances:
                best_instance = self._select_best_instance(active_instances)
                active_general.append(best_instance)
        return active_general[:2]  # Limit to 2 fallback tools

    def _record_selection(
        self, conversation_id: str, user_input: str, selected_tools: list[ToolInstance]
    ) -> None:
        """Record tool selection for learning purposes"""
        if conversation_id not in self.selection_history:
            self.selection_history[conversation_id] = []
        selection_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "user_input": user_input,
            "selected_tools": [
                {
                    "tool_name": tool.tool_definition.name,
                    "instance_id": tool.instance_id,
                    "capabilities": tool.tool_definition.provides_capabilities,
                }
                for tool in selected_tools
                if tool.tool_definition
            ],
        }
        self.selection_history[conversation_id].append(selection_record)
        # Keep only recent selections (last 50)
        if len(self.selection_history[conversation_id]) > 50:
            self.selection_history[conversation_id].pop(0)

    def get_selection_stats(self) -> dict[str, Any]:
        """Get tool selection statistics"""
        total_selections = sum(
            len(history) for history in self.selection_history.values()
        )
        tool_usage_count = {}
        for history in self.selection_history.values():
            for selection in history:
                for tool_info in selection["selected_tools"]:
                    tool_name = tool_info["tool_name"]
                    tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1
        return {
            "total_conversations": len(self.selection_history),
            "total_selections": total_selections,
            "tool_usage_frequency": dict(
                sorted(tool_usage_count.items(), key=lambda x: x[1], reverse=True)
            ),
            "average_tools_per_selection": (
                total_selections / len(self.selection_history)
                if self.selection_history
                else 0.0
            ),
        }


class ToolLifecycleManager:
    """Main tool lifecycle management orchestrator"""

    def __init__(self, comm_hub, response_router, event_bus):  # Assuming these exist
        self.comm_hub = comm_hub
        self.response_router = response_router
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        # Core components
        self.registry = ToolRegistry()
        self.health_monitor = ToolHealthMonitor(self.registry, event_bus)
        self.dependency_resolver = DependencyResolver(self.registry)
        self.tool_selector = IntelligentToolSelector(self.registry, response_router)
        # Lifecycle management
        self._running = False
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the tool lifecycle manager"""
        if self._running:
            return
        self._running = True
        # Start health monitoring
        await self.health_monitor.start()
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Tool Lifecycle Manager started")

    async def stop(self) -> None:
        """Stop the tool lifecycle manager"""
        if not self._running:
            return
        self._running = False
        # Stop health monitoring
        await self.health_monitor.stop()
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        # Deactivate all tools
        await self._deactivate_all_tools()
        self.logger.info("Tool Lifecycle Manager stopped")

    async def register_tool(
        self, definition: ToolDefinition, auto_activate: bool = True
    ) -> str | None:
        """Register and optionally activate a new tool"""
        # Register definition
        if not self.registry.register_tool_definition(definition):
            return None
        # Create instance if auto-activate
        if auto_activate:
            instance = self.registry.create_tool_instance(definition.tool_id)
            if instance:
                await self.activate_tool(instance.instance_id)
                return instance.instance_id
        return definition.tool_id

    async def activate_tool(self, instance_id: str) -> bool:
        """Activate a tool instance"""
        instance = self.registry.get_tool_instance(instance_id)
        if not instance:
            self.logger.error(f"Tool instance not found: {instance_id}")
            return False
        try:
            # Check dependencies
            dep_status = self.dependency_resolver.resolve_dependencies(instance)
            if dep_status == DependencyStatus.UNSATISFIED:
                self.logger.error(
                    f"Dependencies not satisfied for tool: {instance.tool_definition.name}"
                )
                return False
            # Transition to initializing
            if not instance.transition_state(
                ToolState.INITIALIZING, "Starting activation"
            ):
                self.logger.error(f"Cannot activate tool in state: {instance.state}")
                return False
            # Perform activation (placeholder for actual implementation)
            success = await self._perform_activation(instance)
            if success:
                # Transition to active
                instance.transition_state(ToolState.ACTIVE, "Activation successful")
                # Register with communication hub if it's a plugin
                if instance.tool_definition:
                    # This would integrate with actual plugin loading
                    # self.comm_hub.register_plugin(
                    #     instance,  # Placeholder - would be actual plugin instance
                    #     instance.tool_definition.provides_capabilities,
                    #     PluginDependency(
                    #         instance.tool_definition.name,
                    #         instance.tool_definition.requires_capabilities,
                    #         instance.tool_definition.optional_capabilities,
                    #     ),
                    # )
                    pass  # Placeholder for actual plugin registration
                self.logger.info(
                    f"Successfully activated tool: {instance.tool_definition.name}"
                )
                return True
            else:
                instance.transition_state(ToolState.FAILING, "Activation failed")
                return False
        except Exception as e:
            self.logger.error(f"Error activating tool {instance_id}: {e}")
            instance.record_error("activation_error", str(e))
            instance.transition_state(ToolState.FAILING, f"Activation error: {e}")
            return False

    async def deactivate_tool(self, instance_id: str) -> bool:
        """Deactivate a tool instance"""
        instance = self.registry.get_tool_instance(instance_id)
        if not instance:
            return False
        try:
            # Transition to deactivating
            if not instance.transition_state(
                ToolState.DEACTIVATING, "Starting deactivation"
            ):
                self.logger.warning(
                    f"Forced deactivation of tool in state: {instance.state}"
                )
            # Perform deactivation
            success = await self._perform_deactivation(instance)
            # Unregister from communication hub
            if instance.tool_definition:
                # self.comm_hub.unregister_plugin(instance.tool_definition.name)
                pass  # Placeholder for actual plugin unregistration
            # Transition to inactive
            instance.transition_state(
                ToolState.INACTIVE,
                "Deactivation successful" if success else "Deactivation with errors",
            )
            self.logger.info(
                f"Deactivated tool: {instance.tool_definition.name if instance.tool_definition else instance_id}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error deactivating tool {instance_id}: {e}")
            instance.record_error("deactivation_error", str(e))
            return False

    async def process_user_request(
        self, user_input: str, conversation_id: str, context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Process a user request using appropriate tools"""
        try:
            # Select tools for the request
            selected_tools = self.tool_selector.select_tools_for_request(
                user_input, conversation_id, context
            )
            if not selected_tools:
                return {
                    "success": False,
                    "error": "No suitable tools available for this request",
                    "selected_tools": [],
                }

            # Use response router for coordination
            # routing_decision = await self.response_router.route_request(
            #     user_input, conversation_id
            # )
            # Placeholder for routing decision
            class MockRoutingDecision:
                estimated_time = 0.1

            routing_decision = MockRoutingDecision()
            # Execute the routing decision
            # result = await self.response_router.execute_routing_decision(
            #     routing_decision, user_input, conversation_id
            # )
            # Placeholder for execution result
            result = {"echo": user_input, "processed": True}
            # Update tool metrics
            for tool in selected_tools:
                if tool.metrics:
                    success = "error" not in result
                    response_time = routing_decision.estimated_time
                    tool.metrics.record_request(success, response_time)
            return {
                "success": True,
                "result": result,
                "selected_tools": [
                    {
                        "name": tool.tool_definition.name,
                        "instance_id": tool.instance_id,
                        "capabilities": tool.tool_definition.provides_capabilities,
                    }
                    for tool in selected_tools
                    if tool.tool_definition
                ],
                # "routing_decision": {
                #     "strategy": routing_decision.strategy.value,
                #     "confidence": routing_decision.confidence.value,
                #     "reasoning": routing_decision.reasoning,
                # },
                "routing_decision": {
                    "strategy": "direct",  # Placeholder
                    "confidence": "medium",  # Placeholder
                    "reasoning": "Mock routing decision",  # Placeholder
                },
            }
        except Exception as e:
            self.logger.error(f"Error processing user request: {e}")
            return {
                "success": False,
                "error": f"Request processing failed: {e}",
                "selected_tools": [],
            }

    async def _perform_activation(self, instance: ToolInstance) -> bool:
        """Perform the actual tool activation"""
        # Placeholder for actual activation logic
        # In real implementation, this would:
        # 1. Load the plugin class
        # 2. Initialize with configuration
        # 3. Start any necessary processes
        # 4. Verify successful startup
        await asyncio.sleep(0.1)  # Simulate activation time
        return True

    async def _perform_deactivation(self, instance: ToolInstance) -> bool:
        """Perform the actual tool deactivation"""
        # Placeholder for actual deactivation logic
        # In real implementation, this would:
        # 1. Gracefully shutdown processes
        # 2. Clean up resources
        # 3. Save state if necessary
        # 4. Unload plugin
        await asyncio.sleep(0.1)  # Simulate deactivation time
        return True

    async def _deactivate_all_tools(self) -> None:
        """Deactivate all active tools during shutdown"""
        active_instances = self.registry.get_active_instances()
        for instance in active_instances:
            try:
                await self.deactivate_tool(instance.instance_id)
            except Exception as e:
                self.logger.error(
                    f"Error deactivating tool during shutdown: {instance.instance_id}: {e}"
                )

    async def _cleanup_loop(self) -> None:
        """Background cleanup of inactive and destroyed tools"""
        while self._running:
            try:
                # Find tools that need cleanup
                all_instances = list(self.registry.tool_instances.values())
                for instance in all_instances:
                    # Remove destroyed instances
                    if instance.state == ToolState.DESTROYED:
                        self.registry.remove_tool_instance(instance.instance_id)
                        continue
                    # Clean up old inactive instances (older than 1 hour)
                    if (
                        instance.state == ToolState.INACTIVE
                        and instance.metrics
                        and instance.metrics.last_activity
                    ):
                        time_since_activity = (
                            datetime.now(UTC) - instance.metrics.last_activity
                        ).total_seconds()
                        if time_since_activity > 3600:  # 1 hour
                            self.logger.info(
                                f"Cleaning up old inactive tool: {instance.tool_definition.name if instance.tool_definition else instance.instance_id}"
                            )
                            instance.transition_state(
                                ToolState.DESTROYED, "Cleanup due to inactivity"
                            )
                # Sleep before next cleanup cycle
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        registry_stats = self.registry.get_registry_stats()
        # comm_stats = self.comm_hub.get_communication_overview()
        # routing_stats = self.response_router.get_routing_stats()
        selection_stats = self.tool_selector.get_selection_stats()
        return {
            "running": self._running,
            "timestamp": datetime.now(UTC).isoformat(),
            "registry": registry_stats,
            # "communication": comm_stats,
            # "routing": routing_stats,
            "selection": selection_stats,
            "active_tools": len(self.registry.get_active_instances()),
            "healthy_tools": len(
                [
                    instance
                    for instance in self.registry.get_active_instances()
                    if instance.is_healthy()
                ]
            ),
        }


# --- End of Code ---
