"""
Simple in-memory event bus for testing telemetry
"""

import asyncio
from typing import Callable, Dict, List, Union, Any
from ..events import BaseEvent


class SimpleEventBus:
    """Simple in-memory event bus for testing"""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
    
    async def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events of a specific type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(callback)
    
    async def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events of a specific type"""
        if event_type in self.handlers:
            try:
                self.handlers[event_type].remove(callback)
                if not self.handlers[event_type]:
                    del self.handlers[event_type]
            except ValueError:
                pass  # Callback not found, ignore
    
    async def emit_event(self, event: BaseEvent):
        """Emit an event to all subscribers"""
        # Handle wildcard subscriptions (*)
        wildcard_handlers = self.handlers.get("*", [])
        specific_handlers = self.handlers.get(event.event_type, [])
        
        all_handlers = wildcard_handlers + specific_handlers
        
        # Call all handlers
        for handler in all_handlers:
            try:
                await handler(event)
            except Exception as e:
                print(f"Error in event handler: {e}")
    
    async def emit(self, event_or_type: Union[str, "BaseEvent"], **kwargs: Any):
        """Legacy emit method for compatibility"""
        from ..events import create_event, BaseEvent
        
        # If event_or_type is already a BaseEvent, emit it directly
        if isinstance(event_or_type, BaseEvent):
            await self.emit_event(event_or_type)
        else:
            # Add default source_plugin if not provided
            kwargs.setdefault("source_plugin", "SimpleEventBus")
            event = create_event(event_or_type, **kwargs)
            await self.emit_event(event)