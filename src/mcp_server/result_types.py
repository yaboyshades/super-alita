from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(slots=True)
class Result(Generic[T, E]):
    ok: bool
    value: T | None = None
    error: E | None = None

    @classmethod
    def Ok(cls, value: T) -> Result[T, E]:
        return cls(ok=True, value=value)

    @classmethod
    def Err(cls, error: E) -> Result[T, E]:
        return cls(ok=False, error=error)
