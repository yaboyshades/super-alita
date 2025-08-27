"""LangChain adapter fallback definitions."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - executed when LangChain is installed
    from langchain_core.runnables import Runnable  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed when LangChain is missing
    class Runnable:  # type: ignore[misc]
        """Fallback `Runnable` when LangChain is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("LangChain not installed")

        def invoke(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            """Mimic `Runnable.invoke` but always raises an error."""
            raise NotImplementedError("LangChain not installed")
