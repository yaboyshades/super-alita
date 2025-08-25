from __future__ import annotations

import io
from typing import Any

import yaml


def safe_load(s: str | bytes) -> Any:
    """Safe YAML loader wrapper."""
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")
    return yaml.safe_load(io.StringIO(s))


def safe_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False)
