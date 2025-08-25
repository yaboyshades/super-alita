"""Middleware for auth and logging."""
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


async def logging_middleware(handler: Callable, request: Any) -> Any:
    """Log all requests."""
    logger.info(
        f"Request to {request.agent_id} with {len(request.messages)} messages"
    )
    response = await handler(request)
    logger.info(f"Response from {request.agent_id} completed")
    return response


async def auth_middleware(handler: Callable, request: Any) -> Any:
    """Simple API key auth check."""
    if request.messages:
        for part in request.messages[0].parts:
            if getattr(part, "metadata", None) and part.metadata.get("api_key") == "test-key":
                return await handler(request)

    raise ValueError("Missing or invalid API key")
