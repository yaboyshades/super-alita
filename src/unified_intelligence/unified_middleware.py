"""
Unified Intelligence Middleware

Provides middleware layer for integrating unified intelligence
with existing systems and frameworks:
- FastAPI middleware integration
- Event system integration
- Plugin system compatibility
- Graceful error handling and fallbacks
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class UnifiedMiddleware:
    """
    Middleware for unified intelligence integration.

    Provides seamless integration with existing systems while
    maintaining compatibility and graceful degradation.
    """

    def __init__(self, unified_engine=None):
        """Initialize middleware with optional unified engine."""
        self.unified_engine = unified_engine
        self.middleware_enabled = True
        self.request_count = 0
        self.error_count = 0
        self.enhancement_count = 0

    def create_fastapi_middleware(self):
        """Create FastAPI middleware for request enhancement."""

        async def unified_middleware(request, call_next):
            """FastAPI middleware function."""
            if not self.middleware_enabled or not self.unified_engine:
                return await call_next(request)

            try:
                self.request_count += 1

                # Check if this is a relevant endpoint
                if self._should_enhance_request(request):
                    # Extract request data
                    request_data = await self._extract_request_data(request)

                    # Enhance with unified intelligence
                    enhancement = await self._enhance_request(request_data)

                    # Attach enhancement to request
                    if enhancement:
                        request.state.unified_enhancement = enhancement
                        self.enhancement_count += 1

                # Process request
                response = await call_next(request)

                # Enhance response if needed
                if hasattr(request.state, "unified_enhancement"):
                    response = await self._enhance_response(
                        response, request.state.unified_enhancement
                    )

                return response

            except Exception as e:
                logger.warning(f"Unified middleware error: {e}")
                self.error_count += 1
                # Continue without enhancement
                return await call_next(request)

        return unified_middleware

    def _should_enhance_request(self, request) -> bool:
        """Check if request should be enhanced."""
        # Enhance specific endpoints
        enhance_paths = [
            "/v1/chat/stream",
            "/tools/",
            "/ability/execute/",
            "/sdd/",
        ]

        path = getattr(request.url, "path", "")
        return any(enhance_path in path for enhance_path in enhance_paths)

    async def _extract_request_data(self, request) -> dict[str, Any]:
        """Extract relevant data from request."""
        try:
            # Get basic request info
            data = {
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "content_type": request.headers.get("content-type", ""),
            }

            # Extract body for POST requests
            if request.method == "POST":
                if "application/json" in data["content_type"]:
                    try:
                        import json

                        body_bytes = await request.body()
                        if body_bytes:
                            data["body"] = json.loads(body_bytes)
                    except Exception as e:
                        logger.debug(f"Failed to parse JSON body: {e}")
                        data["body"] = None

            return data

        except Exception as e:
            logger.warning(f"Failed to extract request data: {e}")
            return {"error": str(e)}

    async def _enhance_request(
        self, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Enhance request using unified intelligence."""
        if not self.unified_engine:
            return None

        try:
            # Extract user input from common request patterns
            user_input = self._extract_user_input(request_data)
            if not user_input:
                return None

            # Use unified intelligence to enhance
            enhancement = await self.unified_engine.enhance_interaction(
                user_input=user_input, context={"request_data": request_data}
            )

            return enhancement

        except Exception as e:
            logger.warning(f"Request enhancement failed: {e}")
            return None

    def _extract_user_input(self, request_data: dict[str, Any]) -> str | None:
        """Extract user input from request data."""
        body = request_data.get("body")
        if not body:
            return None

        # Common patterns for user input
        input_fields = [
            "message",
            "query",
            "prompt",
            "input",
            "text",
            "question",
            "command",
            "content",
        ]

        for field in input_fields:
            if isinstance(body, dict) and field in body:
                value = body[field]
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    async def _enhance_response(self, response, enhancement: dict[str, Any]):
        """Enhance response with unified intelligence data."""
        try:
            # Add enhancement headers
            if enhancement.get("workflow_detected"):
                response.headers["X-SDD-Workflow"] = enhancement["workflow_detected"]

            if enhancement.get("constitutional_compliance"):
                score = enhancement["constitutional_compliance"].get("overall_score", 0)
                response.headers["X-Constitutional-Score"] = str(score)

            if enhancement.get("mangle_insights"):
                insights_count = len(enhancement["mangle_insights"].get("insights", []))
                response.headers["X-Mangle-Insights"] = str(insights_count)

            # For streaming responses, might need special handling
            if hasattr(response, "streaming") and response.streaming:
                # For streaming, we'll add headers only
                response.headers["X-Unified-Enhancement"] = "true"

            return response

        except Exception as e:
            logger.warning(f"Response enhancement failed: {e}")
            return response

    def create_event_handler(self):
        """Create event handler for event bus integration."""

        async def handle_unified_events(event):
            """Handle events for unified intelligence."""
            if not self.middleware_enabled or not self.unified_engine:
                return

            try:
                event_type = getattr(event, "event_type", None)

                if event_type in ["user_query", "chat_message", "sdd_request"]:
                    # Extract relevant data
                    event_data = self._extract_event_data(event)

                    # Enhance using unified intelligence
                    enhancement = await self.unified_engine.enhance_interaction(
                        user_input=event_data.get("content", ""),
                        context={"event": event_data},
                    )

                    # Emit enhanced event if needed
                    if enhancement:
                        await self._emit_enhancement_event(event, enhancement)

            except Exception as e:
                logger.warning(f"Event handling error: {e}")
                self.error_count += 1

        return handle_unified_events

    def _extract_event_data(self, event) -> dict[str, Any]:
        """Extract relevant data from event."""
        try:
            return {
                "event_type": getattr(event, "event_type", None),
                "content": getattr(event, "content", ""),
                "user_id": getattr(event, "user_id", None),
                "timestamp": getattr(event, "timestamp", None),
                "metadata": getattr(event, "metadata", {}),
            }
        except Exception as e:
            logger.warning(f"Failed to extract event data: {e}")
            return {}

    async def _emit_enhancement_event(self, original_event, enhancement):
        """Emit enhancement event."""
        try:
            # Create enhancement event
            enhancement_event = {
                "event_type": "unified_enhancement",
                "original_event_type": getattr(original_event, "event_type", None),
                "enhancement_data": enhancement,
                "timestamp": self._get_timestamp(),
            }

            # Emit to event bus if available
            if hasattr(original_event, "event_bus"):
                await original_event.event_bus.emit(enhancement_event)

        except Exception as e:
            logger.warning(f"Failed to emit enhancement event: {e}")

    def create_plugin_wrapper(self):
        """Create plugin wrapper for plugin system integration."""

        class UnifiedPlugin:
            """Plugin wrapper for unified intelligence."""

            def __init__(self, middleware_instance):
                self.middleware = middleware_instance
                self.name = "unified_intelligence"

            async def on_message(self, message):
                """Handle plugin messages."""
                if not self.middleware.middleware_enabled:
                    return message

                try:
                    # Extract message content
                    content = getattr(message, "content", str(message))

                    # Enhance using unified intelligence
                    if self.middleware.unified_engine:
                        enhancement = (
                            await self.middleware.unified_engine.enhance_interaction(
                                user_input=content,
                                context={"plugin": "message_handler"},
                            )
                        )

                        # Attach enhancement to message
                        if enhancement:
                            message.unified_enhancement = enhancement

                except Exception as e:
                    logger.warning(f"Plugin message handling error: {e}")

                return message

            async def shutdown(self):
                """Shutdown plugin."""
                logger.info("Unified intelligence plugin shutting down")

        return UnifiedPlugin(self)

    def create_ability_wrapper(self, original_ability: Callable):
        """Wrap an ability with unified intelligence enhancement."""

        async def enhanced_ability(*args, **kwargs):
            """Enhanced ability wrapper."""
            if not self.middleware_enabled or not self.unified_engine:
                return await original_ability(*args, **kwargs)

            try:
                # Extract input from ability arguments
                ability_input = self._extract_ability_input(args, kwargs)

                # Enhance input if relevant
                if ability_input:
                    enhancement = await self.unified_engine.enhance_interaction(
                        user_input=ability_input,
                        context={"ability": original_ability.__name__},
                    )

                    # Add enhancement to kwargs
                    if enhancement:
                        kwargs["unified_enhancement"] = enhancement

                # Execute original ability
                result = await original_ability(*args, **kwargs)

                # Enhance result if needed
                if hasattr(result, "update") and "unified_enhancement" in kwargs:
                    result["unified_enhancement"] = kwargs["unified_enhancement"]

                return result

            except Exception as e:
                logger.warning(f"Ability enhancement error: {e}")
                # Fall back to original ability
                return await original_ability(*args, **kwargs)

        return enhanced_ability

    def _extract_ability_input(self, args: tuple, kwargs: dict) -> str | None:
        """Extract user input from ability arguments."""
        # Check common argument names
        input_keys = ["query", "prompt", "message", "input", "text"]

        for key in input_keys:
            if key in kwargs and isinstance(kwargs[key], str):
                return kwargs[key]

        # Check positional arguments
        for arg in args:
            if isinstance(arg, str) and len(arg.strip()) > 0:
                return arg.strip()

        return None

    def get_middleware_stats(self) -> dict[str, Any]:
        """Get middleware statistics."""
        return {
            "middleware_enabled": self.middleware_enabled,
            "total_requests": self.request_count,
            "enhanced_requests": self.enhancement_count,
            "error_count": self.error_count,
            "enhancement_rate": (
                self.enhancement_count / self.request_count
                if self.request_count > 0
                else 0
            ),
            "error_rate": (
                self.error_count / self.request_count if self.request_count > 0 else 0
            ),
            "unified_engine_available": self.unified_engine is not None,
        }

    def toggle_middleware(self, enabled: bool = True) -> dict[str, Any]:
        """Toggle middleware on/off."""
        self.middleware_enabled = enabled
        return {
            "middleware_enabled": self.middleware_enabled,
            "message": f"Unified middleware {'enabled' if enabled else 'disabled'}",
        }

    def reset_stats(self) -> dict[str, Any]:
        """Reset middleware statistics."""
        old_stats = self.get_middleware_stats()

        self.request_count = 0
        self.error_count = 0
        self.enhancement_count = 0

        return {
            "reset": True,
            "previous_stats": old_stats,
            "message": "Middleware statistics reset",
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        try:
            from datetime import datetime

            return datetime.now().isoformat()
        except Exception:
            return "unknown"

    def get_status(self) -> dict[str, Any]:
        """Get middleware status."""
        return {
            "middleware_enabled": self.middleware_enabled,
            "unified_engine_available": self.unified_engine is not None,
            "stats": self.get_middleware_stats(),
            "capabilities": [
                "fastapi_middleware",
                "event_handling",
                "plugin_integration",
                "ability_enhancement",
                "request_enhancement",
                "response_enhancement",
            ],
        }
