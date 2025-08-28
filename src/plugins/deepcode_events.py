"""DeepCode-specific event metadata helper."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class DeepCodeGenerationEvent:
    repo_path: str
    prompt: str
    context_files: list[str]
    timestamp: datetime | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    def to_atom_metadata(self) -> dict[str, Any]:
        prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16]
        return {
            "event_type": "deepcode_generation",
            "repo_path": self.repo_path,
            "prompt_hash": prompt_hash,
            "context_size": len(self.context_files),
            "timestamp": self.timestamp.isoformat(),
        }
