from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryRecord:
    id: str
    ts: float
    tier: str  # "short" | "long"
    task: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


class AgentVectorDB:
    def __init__(self, root: Path | None = None, short_max: int = 50) -> None:
        self.root = root or Path(os.getenv("AGENT_MEMORY_DIR", "logs/agent_memory")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "memory.jsonl"
        self.short_max = int(os.getenv("AGENT_MEMORY_SHORT_MAX", short_max))

    def _load_all(self) -> list[MemoryRecord]:
        if not self.file.exists():
            return []
        out: list[MemoryRecord] = []
        with self.file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(
                        MemoryRecord(
                            id=obj["id"],
                            ts=float(obj["ts"]),
                            tier=obj.get("tier", "short"),
                            task=obj.get("task", ""),
                            content=obj.get("content", ""),
                            meta=obj.get("meta", {}),
                        )
                    )
                except Exception:
                    continue
        return out

    def _append(self, rec: MemoryRecord) -> None:
        with self.file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def add(self, *, task: str, content: str, meta: dict[str, Any] | None = None, tier: str = "short") -> str:
        rec = MemoryRecord(id=uuid.uuid4().hex, ts=time.time(), tier=tier, task=task, content=content, meta=meta or {})
        self._append(rec)
        return rec.id

    def promote(self, rec_id: str) -> bool:
        items = self._load_all()
        changed = False
        for r in items:
            if r.id == rec_id:
                r.tier = "long"
                changed = True
                break
        if changed:
            tmp = self.file.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for r in items:
                    fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
            tmp.replace(self.file)
        return changed

    def prune_short(self, keep_last: int | None = None) -> int:
        keep = int(keep_last) if keep_last is not None else self.short_max
        items = self._load_all()
        shorts = [r for r in items if r.tier == "short"]
        longs = [r for r in items if r.tier == "long"]
        shorts_sorted = sorted(shorts, key=lambda r: r.ts, reverse=True)
        new_items = shorts_sorted[:keep] + longs
        tmp = self.file.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in sorted(new_items, key=lambda r: r.ts):
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        tmp.replace(self.file)
        return max(0, len(shorts) - keep)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        items = self._load_all()
        if not items:
            return []
        corpus = [r.content for r in items]
        vec = TfidfVectorizer(max_features=4096)
        try:
            mat = vec.fit_transform(corpus)
            qv = vec.transform([query])
            sims = cosine_similarity(qv, mat)[0]
        except Exception:
            return []
        idx = np.argsort(-sims)[: max(1, top_k)]
        results: list[dict[str, Any]] = []
        for i in idx:
            r = items[int(i)]
            results.append({
                "id": r.id,
                "score": float(sims[int(i)]),
                "task": r.task,
                "tier": r.tier,
                "ts": r.ts,
                "meta": r.meta,
                "snippet": r.content[:5000],
            })
        return results

    def summary(self) -> dict[str, Any]:
        items = self._load_all()
        return {
            "total": len(items),
            "short": sum(1 for r in items if r.tier == "short"),
            "long": sum(1 for r in items if r.tier == "long"),
        }
