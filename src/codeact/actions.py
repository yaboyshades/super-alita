"""Action definitions for the CodeAct loop."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CmdRun:
    """Run a shell command."""

    command: Iterable[str]


@dataclass(slots=True)
class IPythonRunCell:
    """Execute a snippet of Python code."""

    code: str


@dataclass(slots=True)
class FileEdit:
    """Edit a file on disk."""

    path: str
    content: str


@dataclass(slots=True)
class BrowseInteractive:
    """Interactive browsing action."""

    url: str
    query: str | None = None


@dataclass(slots=True)
class AgentFinish:
    """Signal that the agent has finished."""

    result: Any | None = None
