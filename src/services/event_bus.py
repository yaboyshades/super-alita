"""Event bus service implementation."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Callable, List
from uuid import uuid4

from .base import BaseService
from ..core.events import create_event

try:
    from reug_runtime.event_bus import make_event_bus, BaseEventBus
except ImportError:
    # Fallback event bus
    class BaseEventBus:
        async def emit(self, event: Dict[str, Any]) -> Dict[str, Any]:
            return event
        
        async def subscribe(self, event_type: str, handler: Callable, source: str = "default") -> None:
            pass
    
    def make_event_bus() -> BaseEventBus:
        return BaseEventBus()

class EventBusService(BaseService):
    """Event bus service with subscription management."""
    
    def __init__(self, config, registry):
        super().__init__(config, registry)
        self.event_bus: BaseEventBus = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
    
    async def initialize(self) -> None:
        """Initialize event bus."""
        try:
            self.event_bus = make_event_bus()
            self._initialized = True
            self.logger.info("Event bus initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize event bus: {e}")
            raise
    
    async def emit(self, event_type: str, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Emit an event with structured format."""
        event = create_event(event_type, **(data or {}), **kwargs)
        
        # Store in history (limited)
        self.event_history.append(event)
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]  # Keep last 500
        
        # Emit through underlying bus
        if hasattr(self.event_bus, 'emit'):
            result = await self.event_bus.emit(event)
        else:
            result = event
        
        # Notify local subscribers
        await self._notify_subscribers(event_type, event)
        
        return result
    
    async def subscribe(self, event_type: str, handler: Callable, source: str = "service") -> str:
        """Subscribe to events."""
        subscription_id = str(uuid4())
        
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        # Wrap handler with error handling
        async def safe_handler(event: Dict[str, Any]):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self.logger.error(f"Event handler error ({event_type}): {e}")
        
        self.subscribers[event_type].append(safe_handler)
        
        self.logger.info(f"Subscribed to {event_type} from {source}")
        return subscription_id
    
    async def _notify_subscribers(self, event_type: str, event: Dict[str, Any]) -> None:
        """Notify all subscribers of an event."""
        handlers = self.subscribers.get(event_type, [])
        if not handlers:
            return
        
        # Run all handlers concurrently
        tasks = [handler(event) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent events for debugging."""
        return self.event_history[-limit:]
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        event_types = {}
        for event in self.event_history:
            event_type = event.get('type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.event_history),
            "event_types": event_types,
            "subscribers": {k: len(v) for k, v in self.subscribers.items()}
        }