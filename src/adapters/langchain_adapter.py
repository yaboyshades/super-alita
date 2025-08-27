
"""
Synchronous wrapper around LangChain's async interface.
The adapter exposes a synchronous ``invoke`` method that delegates to
``ainvoke``. When called from environments that already have an active
``asyncio`` event loop (e.g. notebooks or REPLs), the coroutine is executed on
its own thread to avoid ``RuntimeError`` from ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Protocol


class SupportsAinvoke(Protocol):
    """Protocol for objects exposing an ``ainvoke`` coroutine."""

    async def ainvoke(self, prompt: str) -> Any:  # pragma: no cover - protocol
        ...


class LangChainAdapter:
    """Adapt a LangChain runnable to a simple interface.

    Note:
        In interactive environments where an event loop is already running,
        :meth:`invoke` executes :meth:`ainvoke` in a separate thread with its
        own event loop to avoid conflicts with the active loop.
    """

    def __init__(self, chain: SupportsAinvoke) -> None:
        self._chain = chain

    async def ainvoke(self, prompt: str) -> Any:
        """Asynchronously invoke the underlying chain."""

        return await self._chain.ainvoke(prompt)

    def invoke(self, prompt: str) -> Any:
        """Synchronously invoke the underlying chain."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(prompt))

        result: Any | None = None
        exc: BaseException | None = None

        def runner() -> None:
            nonlocal result, exc
            loop = asyncio.new_event_loop()
            try:
                task = loop.create_task(self.ainvoke(prompt))
                result = loop.run_until_complete(task)
            except BaseException as e:  # pragma: no cover - re-raised below
                exc = e
            finally:
                loop.close()

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()

        if exc is not None:
            raise exc
        return result

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def validate_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Ensure kwargs only contain parameters accepted by ``func``.

    Raises:
        TypeError: If ``kwargs`` contains keys not present in ``func``'s signature.
    """
    signature = inspect.signature(func)
    allowed = set(signature.parameters.keys())
    unexpected = set(kwargs) - allowed
    if unexpected:
        raise TypeError(
            f"{func.__name__}() got unexpected keyword arguments: {sorted(unexpected)}"
        )
    return kwargs


class LangChainAdapter:
    """Simple adapter that validates kwargs before invocation."""

    def invoke(self, func: Callable[..., T], **kwargs: Any) -> T:
        """Call ``func`` ensuring keyword arguments match its signature."""
        validate_kwargs(func, kwargs)
        return func(**kwargs)

