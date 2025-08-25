from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any


def _safe_pow(a: float, b: float) -> float:
    # Basic allowlisted function example
    if abs(a) > 1e6 or abs(b) > 1e3:
        raise ValueError("refuse pathological pow")
    return pow(a, b)


_ALLOW: dict[str, Callable[..., Any]] = {
    "pow": _safe_pow,
}

ALLOWLIST: Mapping[str, Callable[..., Any]] = MappingProxyType(_ALLOW)
