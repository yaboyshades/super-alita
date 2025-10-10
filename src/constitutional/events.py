"""Compliance event envelopes for the constitutional engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class ComplianceEvent:
    """Represents a single constitutional check outcome."""

    article_id: str
    check_id: str
    status: str
    timestamp: datetime
    details: Mapping[str, object]

    def is_pass(self) -> bool:
        return self.status.lower() == "passed"

    def is_fail(self) -> bool:
        return self.status.lower() == "failed"
