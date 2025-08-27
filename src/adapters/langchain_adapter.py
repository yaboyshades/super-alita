from __future__ import annotations

from typing import Any, Callable, TypeVar
import inspect

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
