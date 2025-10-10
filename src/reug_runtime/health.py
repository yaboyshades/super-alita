from __future__ import annotations

"""Shared health checks for runtime dependencies."""

import contextlib
from builtins import anext
from typing import Any


async def check_health(
    event_bus: Any, ability_registry: Any, kg: Any, llm: Any
) -> dict[str, Any]:
    """Ping critical dependencies and return status map.

    Args:
        event_bus: Event emitter used for telemetry.
        ability_registry: Registry responsible for tool contracts.
        kg: Knowledge graph adapter.
        llm: Language model client.

    Returns:
        Mapping with overall ``status`` and per-component details.
    """
    components: dict[str, dict[str, Any]] = {}
    overall = "healthy"

    # Event bus
    try:
        await event_bus.emit({"type": "health_check"})
        components["event_bus"] = {"status": "ok"}
    except Exception as e:  # pragma: no cover - exercised in tests
        components["event_bus"] = {"status": "unhealthy", "error": str(e)}
        overall = "unhealthy"

    # Ability registry
    try:
        if hasattr(ability_registry, "health_check"):
            await ability_registry.health_check({})
        components["ability_registry"] = {"status": "ok"}
    except Exception as e:  # pragma: no cover
        components["ability_registry"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall = "unhealthy"

    # Knowledge graph
    try:
        if hasattr(kg, "retrieve_relevant_context"):
            await kg.retrieve_relevant_context("ping")
        components["kg"] = {"status": "ok"}
    except Exception as e:  # pragma: no cover
        components["kg"] = {"status": "unhealthy", "error": str(e)}
        overall = "unhealthy"

    # LLM
    try:
        if hasattr(llm, "health"):
            llm_health = await llm.health()
            # LLM health returns tuple[bool, str] - (ok, reason)
            if isinstance(llm_health, tuple) and len(llm_health) >= 2:
                is_ok, reason = llm_health[0], llm_health[1]
                if is_ok:
                    components["llm"] = {"status": "ok"}
                else:
                    components["llm"] = {
                        "status": "unhealthy",
                        "error": reason,
                    }
                    overall = "unhealthy"
            elif isinstance(llm_health, dict):
                # Legacy dict format support
                if llm_health.get("status") == "ok":
                    components["llm"] = {"status": "ok"}
                else:
                    components["llm"] = {
                        "status": "unhealthy",
                        "error": llm_health.get("error", "Unknown error"),
                    }
                    overall = "unhealthy"
            else:
                components["llm"] = {
                    "status": "unhealthy",
                    "error": f"Invalid health response format: {type(llm_health)}",
                }
                overall = "unhealthy"
        else:
            # Fallback to old method for backwards compatibility
            stream = llm.stream_chat(
                [{"role": "user", "content": "ping"}], timeout=1
            )
            try:
                await anext(stream)
            finally:
                with contextlib.suppress(Exception):
                    await stream.aclose()
            components["llm"] = {"status": "ok"}
    except Exception as e:  # pragma: no cover
        components["llm"] = {"status": "unhealthy", "error": str(e)}
        overall = "unhealthy"

    return {"status": overall, "components": components}
