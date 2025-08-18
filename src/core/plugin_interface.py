"""
Plugin interface for Super Alita agent.
All plugins must implement this interface for hot-swappable modularity.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import asyncio


class PluginInterface(ABC):
    """
    Abstract base class for all Super Alita plugins.
    
    This interface ensures consistent plugin lifecycle management,
    event handling, and resource cleanup across all agent components.
    """
    
    def __init__(self):
        self.event_bus: Optional[Any] = None
        self.store: Optional[Any] = None
        self.config: Optional[Dict[str, Any]] = None
        self.is_running: bool = False
        self._tasks: list = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifier for this plugin."""
        pass
    
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
    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:
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
        self.is_running = True
    
    async def stop(self) -> None:
        """
        Stop the plugin gracefully.
        
        Default implementation cancels all tasks and calls shutdown.
        Override if you need custom stop behavior.
        """
        self.is_running = False
        
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
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean up plugin resources.
        
        This is where you should:
        - Unsubscribe from events
        - Close connections
        - Save state if needed
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Return health status of the plugin.
        
        Returns:
            Dictionary with health information including:
            - status: "healthy", "degraded", "failed"
            - metrics: Performance metrics
            - issues: List of current issues
        """
        return {
            "plugin": self.name,
            "status": "healthy" if self.is_running else "stopped",
            "version": self.version,
            "metrics": {},
            "issues": []
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
    
    async def emit_event(self, event_type: str, **kwargs) -> None:
        """
        Emit an event to the event bus.
        
        Args:
            event_type: Type of event to emit
            **kwargs: Event data
        """
        if self.event_bus:
            await self.event_bus.emit(event_type, source_plugin=self.name, **kwargs)
    
    async def subscribe(self, event_type: str, handler) -> None:
        """
        Subscribe to events from the event bus.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event
        """
        if self.event_bus:
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
        if self.config:
            return self.config.get(key, default)
        return default
    
    async def get_atom(self, key: str) -> Optional[Any]:
        """
        Get an atom from the neural store.
        
        Args:
            key: Atom key
            
        Returns:
            Atom value or None if not found
        """
        if self.store:
            return await self.store.get(key)
        return None
    
    async def set_atom(self, key: str, value: Any, **metadata) -> None:
        """
        Set an atom in the neural store.
        
        Args:
            key: Atom key
            value: Atom value
            **metadata: Additional metadata for the atom
        """
        if self.store:
            await self.store.set(key, value, **metadata)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
    
    def __repr__(self) -> str:
        return self.__str__()


class BasePlugin(PluginInterface):
    """
    Base plugin implementation with common functionality.
    
    Inherit from this class for plugins that need basic event handling
    and don't require complex initialization.
    """
    
    def __init__(self, name: str, description: str = ""):
        super().__init__()
        self._name = name
        self._description = description or f"Base plugin: {name}"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    async def setup(self, event_bus: Any, store: Any, config: Dict[str, Any]) -> None:
        await super().setup(event_bus, store, config)
    
    async def start(self) -> None:
        await super().start()
    
    async def shutdown(self) -> None:
        pass


# Plugin registry for dynamic plugin loading
_plugin_registry: Dict[str, type] = {}


def register_plugin(plugin_class: type) -> None:
    """Register a plugin class in the global registry."""
    if not issubclass(plugin_class, PluginInterface):
        raise ValueError(f"Plugin {plugin_class} must implement PluginInterface")
    
    # Create a temporary instance to get the name
    temp_instance = plugin_class()
    _plugin_registry[temp_instance.name] = plugin_class


def get_plugin_class(name: str) -> Optional[type]:
    """Get a plugin class by name from the registry."""
    return _plugin_registry.get(name)


def list_registered_plugins() -> Dict[str, type]:
    """Get all registered plugin classes."""
    return _plugin_registry.copy()
