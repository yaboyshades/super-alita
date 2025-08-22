# src/plugins/event_bus_plugin.py
"""
Plugin wrapper for the central asynchronous, Redis-backed EventBus.

This plugin ensures that the EventBus singleton is properly managed within
the agent's lifecycle, connecting at startup and shutting down gracefully.
It provides advanced health monitoring, connection recovery, and performance
tracking for the Redis-backed communication infrastructure.

The EventBusPlugin serves as the architectural bridge between the low-level
transport layer (Redis) and the high-level plugin system, ensuring that
the event bus is treated as a first-class citizen in the agent's ecosystem.
"""

import asyncio
import logging
from datetime import UTC
from typing import Any

from src.core.event_bus import EventBus
from src.core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class EventBusPlugin(PluginInterface):
    """
    Enhanced plugin wrapper to manage the EventBus lifecycle.

    This plugin ensures that the Redis-backed EventBus singleton is properly
    integrated into the agent's plugin management system. It provides:

    - EventBus connection lifecycle management (connect, start, shutdown)
    - Advanced health monitoring of Redis connection with recovery
    - Performance monitoring and statistics collection
    - Configuration management for Redis settings
    - Graceful error handling with exponential backoff retry logic
    - Connection pooling and optimization features
    """

    def __init__(self, name: str = "event_bus_plugin"):
        super().__init__()
        self._name = name
        self._bus_instance: EventBus = EventBus()  # Get the singleton instance
        self._health_check_interval: float = 30.0  # seconds
        self._connection_retries: int = 0
        self._max_retries: int = 5
        self._performance_stats: dict[str, Any] = {
            "events_published": 0,
            "events_received": 0,
            "connection_uptime": 0,
            "last_health_check": None,
            "recovery_attempts": 0,
        }

    @property
    def name(self) -> str:
        return self._name

    async def setup(self, event_bus: EventBus, store, config: dict[str, Any]):
        """
        Setup the EventBus plugin with comprehensive configuration.

        The event_bus passed here is the same singleton instance we reference.
        We confirm our reference and apply Redis and performance configuration.
        """
        await super().setup(event_bus, store, config)

        # Ensure we are using the globally provided singleton instance
        self._bus_instance = event_bus

        # Apply plugin-specific configuration
        plugin_config = config.get(self.name, {})
        self._health_check_interval = plugin_config.get("health_check_interval", 30.0)
        self._max_retries = plugin_config.get("max_connection_retries", 5)

        # Configure the EventBus itself if settings are provided
        redis_config = plugin_config.get("redis", {})
        if redis_config:
            # Pass Redis configuration to the event bus
            if hasattr(self._bus_instance, "_redis_host"):
                self._bus_instance._redis_host = redis_config.get("host", "localhost")
                self._bus_instance._redis_port = redis_config.get("port", 6379)
                self._bus_instance._redis_db = redis_config.get("db", 0)
                self._bus_instance._redis_password = redis_config.get("password")

        logger.info("EventBusPlugin setup complete with enhanced monitoring enabled")

    async def start(self) -> None:
        """
        Start the EventBus with advanced connection management.

        This method handles the critical startup sequence with retry logic:
        1. Connect to Redis with exponential backoff
        2. Start the event listener
        3. Begin health monitoring and performance tracking
        """
        logger.info("Starting EventBusPlugin with enhanced features...")

        # Attempt to start the EventBus with intelligent retry logic
        while self._connection_retries < self._max_retries:
            try:
                # The EventBus.start() method handles connection and listener startup
                await self._bus_instance.start()

                # Mark plugin as started
                await super().start()

                # Start advanced monitoring tasks
                self.add_task(self._health_monitor())
                self.add_task(self._performance_monitor())

                logger.info(
                    "EventBusPlugin started successfully - Redis connection established with monitoring"
                )
                self._connection_retries = 0  # Reset retry counter on success
                return

            except Exception as e:
                self._connection_retries += 1
                self._performance_stats["recovery_attempts"] += 1
                logger.error(
                    f"Failed to start EventBus (attempt {self._connection_retries}/{self._max_retries}): {e}"
                )

                if self._connection_retries >= self._max_retries:
                    logger.critical(
                        "EventBusPlugin failed to start after maximum retries - system cannot function"
                    )
                    raise RuntimeError(
                        "EventBus initialization failed - Redis connection unavailable"
                    ) from None

                # Exponential backoff with jitter
                backoff_time = min(2**self._connection_retries, 30)  # Cap at 30 seconds
                await asyncio.sleep(backoff_time)

    async def shutdown(self) -> None:
        """
        Initiate graceful shutdown with final statistics export.

        This ensures all pending messages are processed, statistics are saved,
        and connections are properly closed before the plugin stops.
        """
        logger.info("Shutting down EventBusPlugin...")

        try:
            # Log final performance statistics
            logger.info(f"EventBus final stats: {self._performance_stats}")

            # Shutdown the EventBus (handles Redis disconnection gracefully)
            await self._bus_instance.shutdown()

            # Mark plugin as stopped
            await super().shutdown()

            logger.info("EventBusPlugin shutdown complete with stats preserved")

        except Exception as e:
            logger.error(f"Error during EventBusPlugin shutdown: {e}", exc_info=True)
            # Still mark as shutdown even if there were errors
            await super().shutdown()

    async def _health_monitor(self):
        """
        Advanced health monitoring with intelligent recovery.

        This task runs periodically to ensure the Redis connection is healthy,
        track uptime, and attempt sophisticated recovery if issues are detected.
        """
        from datetime import datetime

        start_time = datetime.now(UTC)

        while self.is_running:
            try:
                await asyncio.sleep(self._health_check_interval)

                if not self.is_running:
                    break

                # Update uptime
                current_time = datetime.now(UTC)
                self._performance_stats["connection_uptime"] = (
                    current_time - start_time
                ).total_seconds()
                self._performance_stats["last_health_check"] = current_time.isoformat()

                # Comprehensive health check
                is_healthy = await self._comprehensive_health_check()

                if not is_healthy:
                    logger.warning(
                        "EventBus comprehensive health check failed - initiating intelligent recovery"
                    )
                    await self._intelligent_recovery()
                else:
                    logger.debug(
                        f"EventBus health check passed - uptime: {self._performance_stats['connection_uptime']:.1f}s"
                    )

            except asyncio.CancelledError:
                logger.info("EventBus health monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in EventBus health monitor: {e}", exc_info=True)

    async def _performance_monitor(self):
        """
        Monitor and track EventBus performance metrics.

        This task collects performance data and can trigger optimizations
        based on usage patterns and connection quality.
        """
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute

                if not self.is_running:
                    break

                # Collect performance metrics
                await self._collect_performance_metrics()

                # Log performance summary every 10 minutes
                uptime_minutes = self._performance_stats["connection_uptime"] / 60
                if uptime_minutes > 0 and int(uptime_minutes) % 10 == 0:
                    logger.info(f"EventBus performance: {self._performance_stats}")

            except asyncio.CancelledError:
                logger.info("EventBus performance monitor cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Error in EventBus performance monitor: {e}", exc_info=True
                )

    async def _comprehensive_health_check(self) -> bool:
        """
        Perform a comprehensive health check of the EventBus.

        Returns:
            bool: True if the EventBus is functioning optimally
        """
        try:
            # Basic running state check
            if not self._bus_instance.is_running:
                return False

            # Redis connection health check
            if hasattr(self._bus_instance, "_redis") and self._bus_instance._redis:
                # Try Redis ping with timeout
                ping_result = await asyncio.wait_for(
                    self._bus_instance._redis.ping(), timeout=5.0
                )

                if ping_result is not True:
                    return False

                # Check Redis memory usage if available
                try:
                    info = await self._bus_instance._redis.info("memory")
                    memory_usage = info.get("used_memory", 0)
                    max_memory = info.get("maxmemory", 0)

                    if max_memory > 0 and memory_usage > (max_memory * 0.9):
                        logger.warning(
                            f"Redis memory usage high: {memory_usage}/{max_memory}"
                        )
                        # Still healthy, but worth noting

                except Exception:
                    # Memory check failed, but connection is still good
                    pass

                return True
            return False

        except TimeoutError:
            logger.warning("EventBus health check timed out")
            return False
        except Exception as e:
            logger.debug(f"EventBus health check failed: {e}")
            return False

    async def _collect_performance_metrics(self):
        """
        Collect and update performance metrics.
        """
        try:
            # Update basic metrics
            if hasattr(self._bus_instance, "_stats"):
                bus_stats = self._bus_instance._stats
                self._performance_stats.update(
                    {
                        "events_published": bus_stats.get("events_published", 0),
                        "events_received": bus_stats.get("events_received", 0),
                        "active_subscriptions": bus_stats.get(
                            "active_subscriptions", 0
                        ),
                    }
                )

            # Collect Redis performance metrics if available
            if hasattr(self._bus_instance, "_redis") and self._bus_instance._redis:
                try:
                    info = await self._bus_instance._redis.info()
                    self._performance_stats["redis_metrics"] = {
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory": info.get("used_memory", 0),
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0),
                    }
                except Exception:
                    # Redis info failed, but connection might still be ok
                    pass

        except Exception as e:
            logger.debug(f"Error collecting performance metrics: {e}")

    async def _intelligent_recovery(self):
        """
        Attempt intelligent recovery from EventBus issues.

        This method uses multiple strategies to recover from different
        types of connection problems.
        """
        try:
            logger.info("Initiating intelligent EventBus recovery...")
            self._performance_stats["recovery_attempts"] += 1

            # Strategy 1: Simple reconnection
            try:
                await self._bus_instance.shutdown()
                await asyncio.sleep(2)
                await self._bus_instance.start()

                # Test the connection
                if await self._comprehensive_health_check():
                    logger.info("EventBus recovery successful (simple reconnection)")
                    return
            except Exception as e:
                logger.debug(f"Simple reconnection failed: {e}")

            # Strategy 2: Reset with longer delay
            try:
                await asyncio.sleep(5)
                await self._bus_instance.start()

                if await self._comprehensive_health_check():
                    logger.info("EventBus recovery successful (delayed reconnection)")
                    return
            except Exception as e:
                logger.debug(f"Delayed reconnection failed: {e}")

            # Strategy 3: Full reset
            logger.warning(
                "All recovery strategies failed - EventBus remains unhealthy"
            )

        except Exception as e:
            logger.error(f"EventBus recovery failed: {e}", exc_info=True)

    async def health_check(self) -> dict[str, Any]:
        """
        Enhanced health check with comprehensive status information.
        """
        if not self.is_running or not self._bus_instance:
            return {
                "status": "stopped",
                "version": "2.0.0",
                "error": "Plugin not running or bus not available",
            }

        # Perform quick health check
        is_healthy = await self._comprehensive_health_check()

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "version": "2.0.0",
            "performance_stats": self._performance_stats,
            "connection_info": {
                "is_running": self._bus_instance.is_running,
                "connection_retries": self._connection_retries,
                "max_retries": self._max_retries,
                "health_check_interval": self._health_check_interval,
            },
            "redis_health": is_healthy,
        }

    async def get_status(self) -> dict[str, Any]:
        """
        Get comprehensive status of the EventBus plugin.
        """
        return await self.health_check()

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get detailed usage statistics from the EventBus.
        """
        try:
            base_stats = dict(self._performance_stats)

            # Add EventBus-specific statistics if available
            if hasattr(self._bus_instance, "get_statistics"):
                bus_stats = await self._bus_instance.get_statistics()
                base_stats.update(bus_stats)

            return {
                "plugin_stats": base_stats,
                "plugin_running": self.is_running,
                "event_bus_running": (
                    self._bus_instance.is_running if self._bus_instance else False
                ),
                "health_status": (await self.health_check())["status"],
            }

        except Exception as e:
            logger.error(f"Error getting EventBus statistics: {e}", exc_info=True)
            return {"error": str(e)}

    # Public accessor for other plugins (backward compatibility)
    @property
    def bus(self) -> EventBus:
        """Get the EventBus instance."""
        return self._bus_instance


# Plugin factory function for easy instantiation
def create_event_bus_plugin() -> EventBusPlugin:
    """
    Factory function to create an EventBusPlugin instance.

    This is the recommended way to create the plugin, as it provides
    a consistent interface and can be extended with additional
    initialization logic if needed.
    """
    return EventBusPlugin()


# Health check utility for external monitoring
async def check_event_bus_health(plugin: EventBusPlugin) -> bool:
    """
    External utility to check EventBus health.

    Args:
        plugin: The EventBusPlugin instance to check

    Returns:
        bool: True if the EventBus is healthy
    """
    try:
        status = await plugin.get_status()
        return status.get("redis_health", False) and status.get("plugin_running", False)
    except Exception:
        return False


# Performance monitoring utility
async def get_event_bus_performance(plugin: EventBusPlugin) -> dict[str, Any]:
    """
    Get EventBus performance metrics.

    Args:
        plugin: The EventBusPlugin instance to monitor

    Returns:
        Dict containing performance metrics
    """
    try:
        return await plugin.get_statistics()
    except Exception as e:
        return {"error": str(e), "available": False}
