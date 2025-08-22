#!/usr/bin/env python3
"""
Plugin-Agent Communication System
Implements inter-plugin messaging, event routing, and dependency resolution.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from core.events import create_event
from core.plugin_interface import PluginInterface


class MessagePriority(Enum):
    """Message priority levels for routing"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MessageStatus(Enum):
    """Message delivery status"""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class PluginMessage:
    """Inter-plugin message with routing and delivery metadata"""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipients: list[str] = field(default_factory=list)
    message_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    reply_to: str | None = None
    correlation_id: str | None = None
    requires_ack: bool = False
    delivery_attempts: int = 0
    max_attempts: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipients": self.recipients,
            "message_type": self.message_type,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reply_to": self.reply_to,
            "correlation_id": self.correlation_id,
            "requires_ack": self.requires_ack,
            "delivery_attempts": self.delivery_attempts,
            "max_attempts": self.max_attempts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginMessage":
        """Create message from dictionary"""
        # Parse datetime fields
        created_at = datetime.fromisoformat(data["created_at"])
        expires_at = (
            datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None
        )

        return cls(
            message_id=data["message_id"],
            sender=data["sender"],
            recipients=data["recipients"],
            message_type=data["message_type"],
            payload=data["payload"],
            priority=MessagePriority(data["priority"]),
            status=MessageStatus(data["status"]),
            created_at=created_at,
            expires_at=expires_at,
            reply_to=data.get("reply_to"),
            correlation_id=data.get("correlation_id"),
            requires_ack=data["requires_ack"],
            delivery_attempts=data["delivery_attempts"],
            max_attempts=data["max_attempts"],
            metadata=data["metadata"],
        )


@dataclass
class MessageRoute:
    """Routing configuration for message types"""

    message_type: str
    recipients: list[str]
    conditions: dict[str, Any] = field(default_factory=dict)
    transform_func: Callable | None = None
    priority_override: MessagePriority | None = None


class PluginDependency:
    """Represents dependencies between plugins"""

    def __init__(
        self,
        plugin_name: str,
        required_capabilities: list[str],
        optional_capabilities: list[str] = None,
    ):
        self.plugin_name = plugin_name
        self.required_capabilities = required_capabilities or []
        self.optional_capabilities = optional_capabilities or []

    def is_satisfied(self, available_capabilities: set[str]) -> bool:
        """Check if all required capabilities are available"""
        required_set = set(self.required_capabilities)
        return required_set.issubset(available_capabilities)

    def get_available_optional(self, available_capabilities: set[str]) -> list[str]:
        """Get which optional capabilities are available"""
        optional_set = set(self.optional_capabilities)
        return list(optional_set.intersection(available_capabilities))


class PluginCommunicationHub:
    """Central hub for inter-plugin communication and coordination"""

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)

        # Plugin registry and message queues
        self.registered_plugins: dict[str, PluginInterface] = {}
        self.plugin_capabilities: dict[str, set[str]] = {}
        self.plugin_dependencies: dict[str, PluginDependency] = {}
        self.message_queues: dict[str, asyncio.Queue] = {}

        # Routing and delivery
        self.message_routes: dict[str, MessageRoute] = {}
        self.pending_messages: dict[str, PluginMessage] = {}
        self.message_handlers: dict[str, dict[str, Callable]] = {}

        # Status tracking
        self.active_conversations: dict[str, list[str]] = {}
        self.delivery_stats: dict[str, dict[str, int]] = {}

        # Background tasks
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start the communication hub"""
        self._running = True

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._cleanup_expired_messages()),
            asyncio.create_task(self._dependency_monitor()),
        ]

        self.logger.info("Plugin Communication Hub started")

    async def stop(self) -> None:
        """Stop the communication hub"""
        self._running = False

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self.logger.info("Plugin Communication Hub stopped")

    def register_plugin(
        self,
        plugin: PluginInterface,
        capabilities: list[str],
        dependencies: PluginDependency = None,
    ) -> None:
        """Register a plugin with the communication hub"""
        plugin_name = plugin.name

        self.registered_plugins[plugin_name] = plugin
        self.plugin_capabilities[plugin_name] = set(capabilities)
        self.message_queues[plugin_name] = asyncio.Queue()
        self.message_handlers[plugin_name] = {}
        self.delivery_stats[plugin_name] = {"sent": 0, "received": 0, "failed": 0}

        if dependencies:
            self.plugin_dependencies[plugin_name] = dependencies

        self.logger.info(
            f"Registered plugin: {plugin_name} with capabilities: {capabilities}"
        )

        # Emit registration event (simplified for testing)
        try:
            event = create_event(
                "plugin_registered",
                plugin_name=plugin_name,
                capabilities=capabilities,
                dependencies=dependencies.plugin_name if dependencies else None,
                source_plugin="communication_hub",
            )
            asyncio.create_task(self.event_bus.emit(event))
        except Exception as e:
            self.logger.warning(f"Could not emit registration event: {e}")

    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin from the communication hub"""
        if plugin_name in self.registered_plugins:
            del self.registered_plugins[plugin_name]
            del self.plugin_capabilities[plugin_name]
            del self.message_queues[plugin_name]
            del self.message_handlers[plugin_name]
            del self.delivery_stats[plugin_name]

            if plugin_name in self.plugin_dependencies:
                del self.plugin_dependencies[plugin_name]

            self.logger.info(f"Unregistered plugin: {plugin_name}")

    def register_message_handler(
        self, plugin_name: str, message_type: str, handler: Callable
    ) -> None:
        """Register a message handler for a specific message type"""
        if plugin_name not in self.message_handlers:
            self.message_handlers[plugin_name] = {}

        self.message_handlers[plugin_name][message_type] = handler
        self.logger.debug(f"Registered handler for {message_type} in {plugin_name}")

    def add_message_route(self, route: MessageRoute) -> None:
        """Add a message routing rule"""
        self.message_routes[route.message_type] = route
        self.logger.debug(f"Added route for message type: {route.message_type}")

    async def send_message(self, message: PluginMessage) -> bool:
        """Send a message to specified recipients"""
        # Validate message
        if not message.recipients:
            self.logger.error(f"Message {message.message_id} has no recipients")
            return False

        # Apply routing rules
        if message.message_type in self.message_routes:
            route = self.message_routes[message.message_type]

            # Override recipients if specified in route
            if route.recipients:
                message.recipients = route.recipients

            # Apply priority override
            if route.priority_override:
                message.priority = route.priority_override

        # Check dependencies for recipients
        valid_recipients = []
        for recipient in message.recipients:
            if recipient in self.registered_plugins:
                # Check if dependencies are satisfied
                if recipient in self.plugin_dependencies:
                    dep = self.plugin_dependencies[recipient]
                    available_caps = self._get_all_available_capabilities()

                    if dep.is_satisfied(available_caps):
                        valid_recipients.append(recipient)
                    else:
                        self.logger.warning(
                            f"Dependencies not satisfied for {recipient}"
                        )
                else:
                    valid_recipients.append(recipient)
            else:
                self.logger.warning(f"Recipient {recipient} not registered")

        if not valid_recipients:
            self.logger.error(f"No valid recipients for message {message.message_id}")
            return False

        message.recipients = valid_recipients

        # Queue message for delivery
        self.pending_messages[message.message_id] = message

        # Update stats
        if message.sender in self.delivery_stats:
            self.delivery_stats[message.sender]["sent"] += 1

        self.logger.debug(
            f"Queued message {message.message_id} for delivery to {valid_recipients}"
        )
        return True

    async def broadcast_message(
        self,
        sender: str,
        message_type: str,
        payload: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> list[str]:
        """Broadcast a message to all registered plugins"""
        recipients = list(self.registered_plugins.keys())

        # Remove sender from recipients
        if sender in recipients:
            recipients.remove(sender)

        if not recipients:
            return []

        message = PluginMessage(
            sender=sender,
            recipients=recipients,
            message_type=message_type,
            payload=payload,
            priority=priority,
        )

        success = await self.send_message(message)
        return recipients if success else []

    async def _message_processor(self) -> None:
        """Background task to process pending messages"""
        while self._running:
            try:
                # Process pending messages
                messages_to_remove = []

                for message_id, message in self.pending_messages.items():
                    # Check if message expired
                    if message.expires_at and datetime.now(UTC) > message.expires_at:
                        message.status = MessageStatus.EXPIRED
                        messages_to_remove.append(message_id)
                        continue

                    # Attempt delivery
                    delivered = await self._deliver_message(message)

                    if delivered:
                        message.status = MessageStatus.DELIVERED
                        messages_to_remove.append(message_id)
                    else:
                        message.delivery_attempts += 1

                        if message.delivery_attempts >= message.max_attempts:
                            message.status = MessageStatus.FAILED
                            messages_to_remove.append(message_id)

                # Remove processed messages
                for message_id in messages_to_remove:
                    del self.pending_messages[message_id]

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1.0)

    async def _deliver_message(self, message: PluginMessage) -> bool:
        """Deliver a message to its recipients"""
        delivery_success = True

        for recipient in message.recipients:
            try:
                if recipient not in self.message_queues:
                    self.logger.warning(f"No message queue for {recipient}")
                    delivery_success = False
                    continue

                # Add to recipient's queue
                await self.message_queues[recipient].put(message)

                # Call handler if registered
                if (
                    recipient in self.message_handlers
                    and message.message_type in self.message_handlers[recipient]
                ):
                    handler = self.message_handlers[recipient][message.message_type]
                    try:
                        # Call handler asynchronously
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        self.logger.error(f"Handler error for {recipient}: {e}")

                # Update stats
                if recipient in self.delivery_stats:
                    self.delivery_stats[recipient]["received"] += 1

                self.logger.debug(
                    f"Delivered message {message.message_id} to {recipient}"
                )

            except Exception as e:
                self.logger.error(f"Failed to deliver message to {recipient}: {e}")
                delivery_success = False

                if recipient in self.delivery_stats:
                    self.delivery_stats[recipient]["failed"] += 1

        return delivery_success

    async def _cleanup_expired_messages(self) -> None:
        """Background task to clean up expired messages"""
        while self._running:
            try:
                current_time = datetime.now(UTC)
                expired_messages = []

                for message_id, message in self.pending_messages.items():
                    if message.expires_at and current_time > message.expires_at:
                        expired_messages.append(message_id)

                for message_id in expired_messages:
                    del self.pending_messages[message_id]
                    self.logger.debug(f"Cleaned up expired message: {message_id}")

                await asyncio.sleep(60.0)  # Clean up every minute

            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60.0)

    async def _dependency_monitor(self) -> None:
        """Background task to monitor plugin dependencies"""
        while self._running:
            try:
                # Check dependency satisfaction for all plugins
                available_capabilities = self._get_all_available_capabilities()

                for plugin_name, dependency in self.plugin_dependencies.items():
                    is_satisfied = dependency.is_satisfied(available_capabilities)

                    # Log dependency status
                    self.logger.debug(
                        f"Plugin {plugin_name} dependency satisfied: {is_satisfied}"
                    )

                await asyncio.sleep(30.0)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in dependency monitor: {e}")
                await asyncio.sleep(30.0)

    def _get_all_available_capabilities(self) -> set[str]:
        """Get all capabilities available across registered plugins"""
        all_capabilities = set()
        for capabilities in self.plugin_capabilities.values():
            all_capabilities.update(capabilities)
        return all_capabilities

    def get_plugin_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all plugins"""
        stats = {}

        for plugin_name in self.registered_plugins:
            stats[plugin_name] = {
                "capabilities": list(self.plugin_capabilities.get(plugin_name, [])),
                "queue_size": self.message_queues[plugin_name].qsize(),
                "delivery_stats": self.delivery_stats.get(plugin_name, {}),
                "has_dependencies": plugin_name in self.plugin_dependencies,
                "dependency_satisfied": True,  # Will be calculated if dependencies exist
            }

            # Check dependency satisfaction
            if plugin_name in self.plugin_dependencies:
                dependency = self.plugin_dependencies[plugin_name]
                available_caps = self._get_all_available_capabilities()
                stats[plugin_name]["dependency_satisfied"] = dependency.is_satisfied(
                    available_caps
                )

        return stats

    def get_communication_overview(self) -> dict[str, Any]:
        """Get overview of communication system status"""
        return {
            "registered_plugins": len(self.registered_plugins),
            "total_capabilities": len(self._get_all_available_capabilities()),
            "pending_messages": len(self.pending_messages),
            "active_routes": len(self.message_routes),
            "running": self._running,
            "plugin_stats": self.get_plugin_stats(),
        }
