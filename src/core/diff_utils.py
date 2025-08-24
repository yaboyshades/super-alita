from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import hashlib

DiffDict = Dict[str, Any]

@dataclass
class DiffEntry:
    path: str
    change_type: str = "modify"   # modify|add|delete
    unified_diff: str = ""
    new_content: Optional[str] = None
    old_sha: Optional[str] = None
    proposed_by: str = "unknown"
    confidence: float = 0.0

    def to_dict(self) -> DiffDict:
        d = asdict(self)
        if self.new_content is not None:
            d["new_sha"] = hashlib.sha1(self.new_content.encode("utf-8")).hexdigest()
        return d

def normalize_diffs(diffs: List[Dict[str, Any]], *, proposed_by: str) -> List[DiffDict]:
    out: List[DiffDict] = []
    for d in diffs or []:
        entry = DiffEntry(
            path=d.get("path"),
            change_type=d.get("change_type", "modify"),
            unified_diff=d.get("unified_diff") or d.get("diff") or "",
            new_content=d.get("new_content"),
            old_sha=d.get("old_sha"),
            proposed_by=proposed_by,
            confidence=float(d.get("confidence", 0.0)),
        )
        if not entry.path:
            raise ValueError("diff entry missing path")
        out.append(entry.to_dict())
    return out
