"""
MCP Router for Super Alita Integration

Central router for MCP tool invocation with single-flight execution, caching,
circuit breaker, and comprehensive event observability.
"""

import asyncio
import hashlib
import json
from typing import Any

from src.core.clock import MonotonicClock
from src.mcp.clients import MCPClientPool


class MCPRouter:
    """
    Central MCP router with comprehensive integration

    Provides single-flight execution, caching, circuit breaker protection,
    and full event observability for MCP tool invocation.
    """

    def __init__(
        self,
        tlm: Any,
        bus: Any,
        client: MCPClientPool | None = None,
        breaker_factory: Any = None,
        cache: Any = None,
        clock: MonotonicClock | None = None,
    ):
        self.tlm = tlm
        self.bus = bus
        self.client = client or MCPClientPool()
        self.breaker_factory = breaker_factory
        self.cache = cache
        self.clock = clock or MonotonicClock()

        # Single-flight execution tracking
        self._in_flight: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    def _emit_telemetry_event(self, event_type: str, **kwargs: Any) -> None:
        """Emit telemetry event to TypeScript extension via HTTP"""
        try:
            import urllib.parse
            import urllib.request

            payload = {
                "event_type": event_type,
                "timestamp": self.clock.now(),
                **kwargs,
            }

            data = urllib.parse.urlencode({"event": json.dumps(payload)}).encode()
            req = urllib.request.Request(
                "http://localhost:17893/telemetry",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            # Fire and forget - don't block on telemetry
            try:
                urllib.request.urlopen(req, timeout=0.1)
            except Exception:
                pass  # Silently ignore telemetry failures
        except Exception:
            pass  # Silently ignore telemetry setup failures

    def _generate_cache_key(self, tool: str, args: dict[str, Any]) -> str:
        """Generate deterministic cache key"""
        # Sort args to ensure consistent key regardless of order
        sorted_args = json.dumps(args, sort_keys=True, separators=(",", ":"))
        key_data = f"{tool}:{sorted_args}"
        return f"mcp_cache:{hashlib.sha256(key_data.encode()).hexdigest()[:16]}"

    async def invoke(
        self, tool: str, args: dict[str, Any], correlation_id: str | None = None
    ) -> dict[str, Any]:
        """
        Invoke MCP tool with full integration

        Args:
            tool: Name of tool to invoke
            args: Tool arguments
            correlation_id: Session/correlation identifier

        Returns:
            Tool execution result
        """
        # Generate cache key
        cache_key = self._generate_cache_key(tool, args)

        # Check cache first
        if self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                # Emit telemetry for cache hit
                self._emit_telemetry_event(
                    "tool_choice",
                    tool_name=tool,
                    confidence=1.0,
                    fallback_considered=False,
                    why="Cache hit - using cached result",
                )

                await self.bus.emit(
                    "tool_invocation_cached",
                    tool=tool,
                    correlation_id=correlation_id,
                    cache_key=cache_key,
                )
                return cached_result

        # Single-flight protection
        async with self._lock:
            if cache_key in self._in_flight:
                # Wait for in-flight execution to complete
                result = await self._in_flight[cache_key]
                return result

            # Create new execution future
            future: asyncio.Future[dict[str, Any]] = asyncio.Future()
            self._in_flight[cache_key] = future

        # Now execute the actual logic (this happens only once per cache_key)
        try:
            result = await self._execute_tool_logic(tool, args, correlation_id)
            future.set_result(result)
            return result
        except Exception as e:
            # Only set exception if future is not already done
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            # Clean up in-flight tracking
            async with self._lock:
                if cache_key in self._in_flight:
                    del self._in_flight[cache_key]

    async def _execute_tool_logic(
        self, tool: str, args: dict[str, Any], correlation_id: str | None = None
    ) -> dict[str, Any]:
        """Execute the actual tool logic - separated for single-flight pattern"""
        try:
            # Get instance from TLM
            instance_info = await self.tlm.select_instance(tool)
            if not instance_info:
                error_msg = f"No instance available for tool: {tool}"
                # Emit shed decision telemetry
                self._emit_telemetry_event(
                    "shed_decision",
                    reason=f"No instance available for tool: {tool}",
                    tool_name=tool,
                )
                raise ValueError(error_msg)

            instance_id, base_url = instance_info

            # Emit tool choice telemetry
            self._emit_telemetry_event(
                "tool_choice",
                tool_name=tool,
                confidence=0.9,
                fallback_considered=False,
                why=f"Selected instance {instance_id} for tool execution",
            )

            # Emit started event
            start_time = self.clock.now()
            await self.bus.emit(
                "tool_invocation_started",
                tool=tool,
                instance_id=instance_id,
                correlation_id=correlation_id,
                args=args,
            )

            # Execute through circuit breaker (simplified)
            if self.breaker_factory:
                breaker = self.breaker_factory(f"mcp_{tool}")
                try:
                    # Check if breaker allows execution
                    if hasattr(breaker, "should_open") and breaker.should_open:
                        error_msg = "Circuit breaker is open"
                        # Emit breaker state change telemetry
                        self._emit_telemetry_event(
                            "breaker_state_change",
                            from_state="closed",
                            to_state="open",
                            failure_count=getattr(breaker, "failure_count", 0),
                            why="Circuit breaker opened due to failure threshold",
                        )
                        raise Exception(error_msg)
                except Exception as e:
                    if "Circuit breaker is open" in str(e):
                        raise
                    # Other breaker exceptions, continue processing

            # Get MCP client and execute tool
            client = await self.client.get_client(base_url)
            result = await client.call_tool(base_url, tool, args)

            # Calculate duration
            end_time = self.clock.now()
            duration_ms = (end_time - start_time) * 1000

            # Cache successful results
            if self.cache and result.get("success", False):
                cache_key = self._generate_cache_key(tool, args)
                await self.cache.set(cache_key, result, ttl=300.0)  # 5 min TTL

            # Emit completed event
            await self.bus.emit(
                "tool_invocation_completed",
                tool=tool,
                instance_id=instance_id,
                correlation_id=correlation_id,
                result=result,
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            # Emit failed event
            await self.bus.emit(
                "tool_invocation_failed",
                tool=tool,
                correlation_id=correlation_id,
                error=str(e),
            )
            raise

    async def shutdown(self) -> None:
        """Cleanup router resources"""
        if self.client:
            await self.client.shutdown()  # type: ignore

        # Cancel any in-flight tasks
        async with self._lock:
            for future in self._in_flight.values():
                if not future.done():
                    future.cancel()
            self._in_flight.clear()
