"""
Event Type Registry for Super Alita.

Provides versioned event descriptors, schema validation, and canonical event definitions.
Replaces scattered string literals with centralized event type management.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EventDescriptor:
    """
    Describes a registered event type with version and schema information.
    """

    name: str
    version: int
    schema: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    deprecated: bool = False
    successor: str | None = None  # Points to replacement event if deprecated


# Global event registry
EVENT_REGISTRY: dict[str, EventDescriptor] = {}


def register_event(descriptor: EventDescriptor) -> bool:
    """
    Register an event type descriptor.

    Args:
        descriptor: Event descriptor to register

    Returns:
        True if registered (new or upgrade), False if ignored (downgrade)
    """
    existing = EVENT_REGISTRY.get(descriptor.name)

    if existing is None:
        # New event type
        EVENT_REGISTRY[descriptor.name] = descriptor
        logger.debug(
            f"Registered new event type: {descriptor.name} v{descriptor.version}"
        )
        return True

    if descriptor.version > existing.version:
        # Upgrade
        EVENT_REGISTRY[descriptor.name] = descriptor
        logger.info(
            f"Upgraded event type: {descriptor.name} v{existing.version} -> v{descriptor.version}"
        )
        return True

    if descriptor.version == existing.version:
        # Same version, update if different (idempotent)
        if existing != descriptor:
            EVENT_REGISTRY[descriptor.name] = descriptor
            logger.debug(f"Updated event type: {descriptor.name} v{descriptor.version}")
            return True
        return False

    # Downgrade attempt - ignore
    logger.warning(
        f"Ignoring downgrade attempt for event type: {descriptor.name} "
        f"v{descriptor.version} (current: v{existing.version})"
    )
    return False


def get_event(name: str) -> EventDescriptor | None:
    """
    Get event descriptor by name.

    Args:
        name: Event type name

    Returns:
        Event descriptor or None if not found
    """
    return EVENT_REGISTRY.get(name)


def list_events() -> dict[str, EventDescriptor]:
    """
    Get all registered event types.

    Returns:
        Copy of the event registry
    """
    return EVENT_REGISTRY.copy()


def validate_event_payload(event_name: str, payload: dict[str, Any]) -> bool:
    """
    Validate event payload against registered schema.

    Args:
        event_name: Name of event type
        payload: Event payload to validate

    Returns:
        True if valid, False if invalid or no schema defined
    """
    descriptor = get_event(event_name)
    if not descriptor or not descriptor.schema:
        # No schema defined - allow anything
        return True

    # Basic schema validation (could be extended with jsonschema)
    required_fields = descriptor.schema.get("required", [])
    for field in required_fields:
        if field not in payload:
            logger.warning(f"Missing required field '{field}' in {event_name} event")
            return False

    return True


def deprecate_event(name: str, successor: str | None = None) -> bool:
    """
    Mark an event type as deprecated.

    Args:
        name: Event type name to deprecate
        successor: Optional replacement event type name

    Returns:
        True if deprecated, False if event not found
    """
    descriptor = get_event(name)
    if not descriptor:
        return False

    # Create new descriptor with deprecation flag
    deprecated_descriptor = EventDescriptor(
        name=descriptor.name,
        version=descriptor.version,
        schema=descriptor.schema,
        description=descriptor.description,
        deprecated=True,
        successor=successor,
    )

    EVENT_REGISTRY[name] = deprecated_descriptor
    logger.info(
        f"Deprecated event type: {name}"
        + (f" (successor: {successor})" if successor else "")
    )
    return True


# Seed the registry with core event types
def _seed_core_events():
    """Initialize registry with essential Super Alita event types."""

    # Tool execution events
    register_event(
        EventDescriptor(
            name="tool_call",
            version=2,
            schema={
                "type": "object",
                "required": [
                    "tool_call_id",
                    "tool_name",
                    "arguments",
                    "conversation_id",
                ],
                "properties": {
                    "tool_call_id": {"type": "string"},
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "conversation_id": {"type": "string"},
                    "timestamp": {"type": "number"},
                },
            },
            description="Request to execute a registered tool",
        )
    )

    register_event(
        EventDescriptor(
            name="tool_result",
            version=2,
            schema={
                "type": "object",
                "required": ["tool_call_id", "success"],
                "properties": {
                    "tool_call_id": {"type": "string"},
                    "success": {"type": "boolean"},
                    "result": {},  # Any type
                    "error": {"type": "string"},
                    "conversation_id": {"type": "string"},
                    "execution_time_ms": {"type": "number"},
                },
            },
            description="Result of tool execution (success or failure)",
        )
    )

    # Planning events
    register_event(
        EventDescriptor(
            name="goal_received",
            version=1,
            schema={
                "type": "object",
                "required": ["goal", "conversation_id"],
                "properties": {
                    "goal": {"type": "string"},
                    "conversation_id": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                    },
                },
            },
            description="New goal received for processing",
        )
    )

    register_event(
        EventDescriptor(
            name="plan_created",
            version=1,
            schema={
                "type": "object",
                "required": ["plan", "conversation_id"],
                "properties": {
                    "plan": {"type": "object"},
                    "conversation_id": {"type": "string"},
                    "planner": {"type": "string"},
                },
            },
            description="Execution plan created by planner",
        )
    )

    # Memory events
    register_event(
        EventDescriptor(
            name="memory_store",
            version=1,
            schema={
                "type": "object",
                "required": ["key", "value"],
                "properties": {
                    "key": {"type": "string"},
                    "value": {},
                    "ttl": {"type": "number"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
            description="Store data in memory system",
        )
    )

    # System events
    register_event(
        EventDescriptor(
            name="plugin_loaded",
            version=1,
            schema={
                "type": "object",
                "required": ["plugin_name"],
                "properties": {
                    "plugin_name": {"type": "string"},
                    "plugin_class": {"type": "string"},
                    "load_time_ms": {"type": "number"},
                },
            },
            description="Plugin successfully loaded and initialized",
        )
    )

    register_event(
        EventDescriptor(
            name="error_occurred",
            version=1,
            schema={
                "type": "object",
                "required": ["error_type", "message"],
                "properties": {
                    "error_type": {"type": "string"},
                    "message": {"type": "string"},
                    "plugin": {"type": "string"},
                    "context": {"type": "object"},
                },
            },
            description="Error occurred during system operation",
        )
    )


# Initialize core events on module import
_seed_core_events()
logger.info(f"Event registry initialized with {len(EVENT_REGISTRY)} core event types")
