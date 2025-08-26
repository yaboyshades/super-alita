"""
Redis Event Bus Implementation for Super Alita Agent System

Production-grade event bus using Redis for distributed agent communication.
Supports pub/sub, event persistence, and scalable event handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import asdict

import redis.asyncio as redis
from redis.asyncio import Redis, ConnectionPool

from ..events import BaseEvent


logger = logging.getLogger(__name__)


class RedisEventBus:
    """
    Redis-backed event bus for Super Alita Agent system.
    
    Provides:
    - Asynchronous pub/sub messaging
    - Event persistence and replay
    - Distributed agent communication
    - Pattern-based subscriptions
    - Event filtering and routing
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "super_alita",
        event_ttl: int = 86400,  # 24 hours
        max_retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Redis event bus.
        
        Args:
            redis_url: Redis connection URL
            channel_prefix: Prefix for all Redis channels
            event_ttl: Time-to-live for stored events (seconds)
            max_retry_attempts: Maximum retry attempts for failed operations
            retry_delay: Delay between retry attempts
        """
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.event_ttl = event_ttl
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        
        self._redis: Optional[Redis] = None
        self._pubsub: Optional[redis.PubSub] = None
        self._connection_pool: Optional[ConnectionPool] = None
        
        self._subscribers: Dict[str, List[Callable]] = {}
        self._pattern_subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        self._subscription_task: Optional[asyncio.Task] = None
        
        # Event statistics
        self._events_published = 0
        self._events_received = 0
        self._connection_errors = 0
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._connection_pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            
            self._redis = Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self._redis.ping()
            
            self._pubsub = self._redis.pubsub()
            
            logger.info(f"Connected to Redis at {self.redis_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connection_errors += 1
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        try:
            self._running = False
            
            if self._subscription_task:
                self._subscription_task.cancel()
                try:
                    await self._subscription_task
                except asyncio.CancelledError:
                    pass
            
            if self._pubsub:
                await self._pubsub.close()
            
            if self._redis:
                await self._redis.close()
            
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            logger.info("Disconnected from Redis")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
    
    def _get_channel_name(self, event_type: str) -> str:
        """Get Redis channel name for event type."""
        return f"{self.channel_prefix}:events:{event_type}"
    
    def _get_stream_name(self, event_type: str) -> str:
        """Get Redis stream name for event persistence."""
        return f"{self.channel_prefix}:stream:{event_type}"
    
    async def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to the Redis event bus.
        
        Args:
            event: Event to emit
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        
        try:
            # Serialize event
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "source_plugin": event.source_plugin,
                "data": event.metadata
            }
            event_json = json.dumps(event_data, default=str)
            
            # Get channel and stream names
            channel = self._get_channel_name(event.event_type)
            stream = self._get_stream_name(event.event_type)
            
            # Use pipeline for atomic operations
            async with self._redis.pipeline() as pipe:
                # Publish to channel (for real-time subscribers)
                await pipe.publish(channel, event_json)
                
                # Add to stream (for persistence and replay)
                await pipe.xadd(
                    stream,
                    event_data,
                    maxlen=1000,  # Keep last 1000 events per type
                    approximate=True
                )
                
                # Set TTL on stream
                await pipe.expire(stream, self.event_ttl)
                
                # Execute pipeline
                await pipe.execute()
            
            self._events_published += 1
            logger.debug(f"Published event {event.event_id} to channel {channel}")
            
        except Exception as e:
            logger.error(f"Failed to emit event {event.event_id}: {e}")
            await self._handle_connection_error()
            raise
    
    async def subscribe(self, event_type: str, handler: Callable[[BaseEvent], None]) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Function to call when events are received
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        
        # Start subscription task if not running
        if not self._running:
            await self._start_subscription_loop()
        
        # Subscribe to Redis channel
        if self._pubsub:
            channel = self._get_channel_name(event_type)
            await self._pubsub.subscribe(channel)
            logger.info(f"Subscribed to events of type '{event_type}'")
    
    async def subscribe_pattern(self, pattern: str, handler: Callable[[BaseEvent], None]) -> None:
        """
        Subscribe to events matching a pattern.
        
        Args:
            pattern: Pattern to match event types (Redis pattern syntax)
            handler: Function to call when matching events are received
        """
        if pattern not in self._pattern_subscribers:
            self._pattern_subscribers[pattern] = []
        
        self._pattern_subscribers[pattern].append(handler)
        
        # Start subscription task if not running
        if not self._running:
            await self._start_subscription_loop()
        
        # Subscribe to Redis pattern
        if self._pubsub:
            channel_pattern = self._get_channel_name(pattern)
            await self._pubsub.psubscribe(channel_pattern)
            logger.info(f"Subscribed to event pattern '{pattern}'")
    
    async def unsubscribe(self, event_type: str, handler: Callable[[BaseEvent], None]) -> None:
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            handler: Handler function to remove
        """
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                
                # Remove empty subscriber lists
                if not self._subscribers[event_type]:
                    del self._subscribers[event_type]
                    
                    # Unsubscribe from Redis channel if no more handlers
                    if self._pubsub:
                        channel = self._get_channel_name(event_type)
                        await self._pubsub.unsubscribe(channel)
                        logger.info(f"Unsubscribed from events of type '{event_type}'")
    
    async def get_event_history(
        self, 
        event_type: str, 
        limit: int = 100, 
        start_time: Optional[str] = None
    ) -> List[BaseEvent]:
        """
        Get event history from Redis streams.
        
        Args:
            event_type: Type of events to retrieve
            limit: Maximum number of events to return
            start_time: Start time for event retrieval (Redis stream ID format)
            
        Returns:
            List of historical events
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")
        
        try:
            stream = self._get_stream_name(event_type)
            
            # Default to reading from beginning
            start_id = start_time or "-"
            
            # Read from stream
            results = await self._redis.xrevrange(stream, max="+", min=start_id, count=limit)
            
            events = []
            for stream_id, fields in results:
                try:
                    # Reconstruct event
                    event_data = {k.decode(): v.decode() for k, v in fields.items()}
                    
                    from ..events import deserialize_event
                    event = deserialize_event(event_data)
                    events.append(event)
                    
                except Exception as e:
                    logger.warning(f"Failed to deserialize event from stream: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get event history for {event_type}: {e}")
            return []
    
    async def _start_subscription_loop(self) -> None:
        """Start the Redis subscription message loop."""
        if self._running:
            return
        
        self._running = True
        self._subscription_task = asyncio.create_task(self._subscription_loop())
    
    async def _subscription_loop(self) -> None:
        """Main subscription loop for processing Redis messages."""
        if not self._pubsub:
            return
        
        try:
            while self._running:
                try:
                    # Get message with timeout
                    message = await asyncio.wait_for(
                        self._pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0
                    )
                    
                    if message and message["type"] == "message":
                        await self._handle_message(message)
                    elif message and message["type"] == "pmessage":
                        await self._handle_pattern_message(message)
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in subscription loop: {e}")
                    await self._handle_connection_error()
                    await asyncio.sleep(self.retry_delay)
        
        except asyncio.CancelledError:
            logger.info("Subscription loop cancelled")
        except Exception as e:
            logger.error(f"Subscription loop failed: {e}")
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle regular subscription message."""
        try:
            # Deserialize event
            event_data = json.loads(message["data"])
            from ..events import deserialize_event
            event = deserialize_event(event_data)
            
            # Extract event type from channel
            channel = message["channel"].decode()
            event_type = channel.split(":")[-1]
            
            # Call handlers
            if event_type in self._subscribers:
                for handler in self._subscribers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Handler error for event {event.event_id}: {e}")
            
            self._events_received += 1
            
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
    
    async def _handle_pattern_message(self, message: Dict[str, Any]) -> None:
        """Handle pattern subscription message."""
        try:
            # Similar to regular message but with pattern matching
            event_data = json.loads(message["data"])
            from ..events import deserialize_event
            event = deserialize_event(event_data)
            
            # Extract pattern from message
            pattern = message["pattern"].decode()
            
            # Call pattern handlers
            if pattern in self._pattern_subscribers:
                for handler in self._pattern_subscribers[pattern]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Pattern handler error for event {event.event_id}: {e}")
            
            self._events_received += 1
            
        except Exception as e:
            logger.error(f"Failed to handle pattern message: {e}")
    
    async def _handle_connection_error(self) -> None:
        """Handle Redis connection errors with retry logic."""
        self._connection_errors += 1
        
        for attempt in range(self.max_retry_attempts):
            try:
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
                # Try to reconnect
                await self.disconnect()
                await self.connect()
                
                logger.info(f"Reconnected to Redis after {attempt + 1} attempts")
                return
                
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error("Failed to reconnect to Redis after maximum attempts")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "events_published": self._events_published,
            "events_received": self._events_received,
            "connection_errors": self._connection_errors,
            "subscribers": len(self._subscribers),
            "pattern_subscribers": len(self._pattern_subscribers),
            "is_connected": self._redis is not None,
            "is_running": self._running
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self._redis:
                return {"status": "disconnected", "healthy": False}
            
            # Test Redis connection
            start_time = time.time()
            await self._redis.ping()
            latency = time.time() - start_time
            
            stats = self.get_statistics()
            
            return {
                "status": "healthy",
                "healthy": True,
                "latency_ms": latency * 1000,
                "statistics": stats
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e)
            }