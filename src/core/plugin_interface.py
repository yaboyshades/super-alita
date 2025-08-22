"""
Plugin interface for Super Alita agent.
All plugins must implement this interface for hot-swappable modularity.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

# Import the event factory
from src.core.events import create_event

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """
    Abstract base class for all Super Alita plugins.

    This interface ensures consistent plugin lifecycle management,
    event handling, and resource cleanup across all agent components.
    """

    def __init__(self):
        self.event_bus: Any | None = None
        self.store: Any | None = None
        self.config: dict[str, Any] | None = None
        self._is_running: bool = False
        self._tasks: list = []

    @property
    def is_running(self) -> bool:
        """Check if the plugin is currently running."""
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        """Set the running state of the plugin."""
        self._is_running = value

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this plugin."""
        ...

    @property
    def version(self) -> str:
        """Return the version of this plugin."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Return a description of what this plugin does."""
        return f"Plugin: {self.name}"

    @property
    def dependencies(self) -> list:
        """Return list of required plugin dependencies."""
        return []

    @abstractmethod
    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """
        Initialize the plugin with required dependencies.

        Args:
            event_bus: The async event bus for inter-plugin communication
            store: The neural atom store for state management
            config: Plugin-specific configuration
        """
        self.event_bus = event_bus
        self.store = store
        self.config = config

    @abstractmethod
    async def start(self) -> None:
        """
        Start the plugin's main operations.

        This is where you should:
        - Register event listeners
        - Start background tasks
        - Initialize plugin-specific resources
        """
        self._is_running = True

    async def stop(self) -> None:
        """
        Stop the plugin gracefully.

        Default implementation cancels all tasks and calls shutdown.
        Override if you need custom stop behavior.
        """
        self._is_running = False

        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._tasks.clear()
        await self.shutdown()

    async def shutdown(self) -> None:
        """
        Clean up plugin resources.

        This is where you should:
        - Unsubscribe from events
        - Close connections
        - Save state
        """
        # Default implementation - plugins should override if needed

    async def health_check(self) -> dict[str, Any]:
        """
        Return health status for this plugin.

        Returns:
            Dictionary with health information
        """
        return {
            "plugin": self.name,
            "status": "healthy" if self._is_running else "stopped",
            "version": self.version,
            "is_running": self._is_running,
            "tasks": len(self._tasks),
            "metrics": {},
            "issues": [],
        }

    def add_task(self, coro) -> asyncio.Task:
        """
        Add a background task to the plugin's task list.

        This ensures proper cleanup when the plugin is stopped.

        Args:
            coro: Coroutine to run as a background task

        Returns:
            The created asyncio Task
        """
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task

    @property
    def is_running(self) -> bool:
        """Whether the plugin is currently running."""
        return self._is_running

    async def emit_event(self, event_type: str, **kwargs) -> None:
        """
        Emit an event to the event bus with enhanced features.

        This helper method creates a proper Pydantic event object and publishes it.
        Enhanced with genealogy tracking and semantic embedding support.

        Args:
            event_type: Type of event to emit
            **kwargs: Event data
        """
        if not self.event_bus:
            return
        try:
            kwargs.setdefault("source_plugin", self.name)

            if hasattr(self, "neural_atom") and self.neural_atom:
                kwargs.setdefault(
                    "genealogy_depth", getattr(self.neural_atom, "depth", 0)
                )
                kwargs.setdefault(
                    "darwin_godel_signature", getattr(self.neural_atom, "signature", "")
                )

            if "content" in kwargs and self.store and hasattr(self.store, "embed_text"):
                try:
                    embedding = await self.store.embed_text(kwargs["content"])
                    if embedding:
                        kwargs.setdefault("embedding", embedding)
                except Exception as e:  # pragma: no cover
                    logger.debug(f"Could not generate embedding for event: {e}")

            if hasattr(self.event_bus, "publish"):
                event_object = create_event(event_type, **kwargs)
                await self.event_bus.publish(event_object)
            elif hasattr(self.event_bus, "emit"):
                await self.event_bus.emit(event_type, **kwargs)
            else:  # pragma: no cover
                raise RuntimeError("Event bus has neither publish nor emit method")

            logger.debug(f"Plugin '{self.name}' emitted event '{event_type}'")
        except Exception as e:
            self.log("error", f"Failed to emit event '{event_type}': {e}")

    async def _generate_embedding(self, content: str) -> list[float] | None:
        """Generate semantic embedding for content if possible."""
        try:
            # Try to use semantic memory plugin if available
            if (
                hasattr(self, "store")
                and self.store
                and hasattr(self.store, "embed_text")
            ):
                return await self.store.embed_text(content)
        except Exception:
            pass
        return None

    async def subscribe(self, event_type: str, handler) -> None:
        """
        Subscribe to events from the event bus or workspace.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event
        """
        # Try workspace first (unified architecture)
        if (
            hasattr(self, "workspace")
            and self.workspace
            and hasattr(self.workspace, "subscribe")
        ):
            self.workspace.subscribe(event_type, handler)
        # Fallback to event bus (legacy architecture)
        elif self.event_bus:
            await self.event_bus.subscribe(event_type, handler)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value for this plugin.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self.config and key in self.config:
            return self.config[key]
        return default

    def log(self, level: str, message: str, **kwargs) -> None:
        """
        Log a message with plugin context.

        Args:
            level: Log level (info, warning, error, debug)
            message: Log message
            **kwargs: Additional context
        """
        import logging

        logger = logging.getLogger(f"plugin.{self.name}")
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{self.name}] {message}", extra=kwargs)

    # Tool interface methods for REUG execution flow integration
    def get_tools(self) -> list[dict[str, Any]] | None:
        """
        Get available tools from this plugin.

        Returns:
            List of tool dictionaries with name, description, etc.
            None if plugin has no tools
        """
        # Default implementation - plugins can override
        return None

    async def process_request(
        self, user_input: str, memory_context: dict[str, Any]
    ) -> Any:
        """
        Process a request through this plugin.

        Args:
            user_input: User's input message
            memory_context: Context from memory systems

        Returns:
            Plugin-specific response
        """
        # Default implementation - plugins should override if they handle requests
        return {
            "plugin": self.name,
            "status": "no_handler",
            "message": f"Plugin {self.name} does not implement request processing",
        }
