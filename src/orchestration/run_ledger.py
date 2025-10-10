from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from .canonical_events import CanonicalEvent
from .event_sanitizer import sanitize_event_for_ledger


class RunLedgerWriter:
    """Append-only NDJSON ledger for canonical orchestration events."""

    def __init__(
        self, path: str | Path, *, enable_shadow: bool = False
    ) -> None:
        self.path = Path(path)
        self.enable_shadow = enable_shadow
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
            self._ensure_permissions()

    def append(self, event: CanonicalEvent) -> None:
        record = sanitize_event_for_ledger(event)
        line = json.dumps(record, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
        self._ensure_permissions()

    def _ensure_permissions(self) -> None:
        if os.name == "nt":  # Windows ACLs handled separately
            return
        with contextlib.suppress(PermissionError):
            os.chmod(self.path, 0o600)


__all__ = ["RunLedgerWriter"]
