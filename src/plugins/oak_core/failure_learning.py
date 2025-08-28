from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOG_FILE = Path("logs/failure_episodes.jsonl")


@dataclass
class FailureEvent:
    timestamp: float
    component: str
    tool: str | None
    error: str
    context: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp,
            "component": self.component,
            "tool": self.tool,
            "error": self.error,
            "context": self.context,
        })


def record_failure(component: str, error: str, tool: str | None = None, **context: Any) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    evt = FailureEvent(timestamp=time.time(), component=component, tool=tool, error=error.strip(), context=context)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(evt.to_json() + "\n")


def _load_events(limit: int | None = None) -> list[dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    out: list[dict[str, Any]] = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                out.append(json.loads(line))
                if limit and i >= limit:
                    break
            except json.JSONDecodeError:
                continue
    return out


def cluster_failures(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    events = events or _load_events()
    clusters: dict[str, dict[str, Any]] = {}
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        tool = e.get("tool") or "unknown"
        # naive signature: tool + first 80 chars of error
        signature = f"{tool}|{(e.get('error','')[:80]).strip()}"
        by_key[signature].append(e)

    for sig, items in by_key.items():
        tool, _ = sig.split("|", 1)
        clusters[sig] = {
            "tool": tool,
            "count": len(items),
            "last_seen": max(i.get("timestamp", 0) for i in items),
            "example": items[-1],
        }
    return {"total_events": len(events), "clusters": clusters}


def propose_resilience_patches(summary: dict[str, Any] | None = None) -> list[str]:
    summary = summary or cluster_failures()
    proposals: list[str] = []
    for sig, info in summary.get("clusters", {}).items():
        tool = info.get("tool") or "unknown"
        count = info.get("count", 0)
        if count < 3:
            continue  # require repetition
        # Heuristic proposals
        if "timeout" in sig.lower():
            proposals.append(
                f"For goals involving tool `{tool}`, prepend a health-check node and increase timeout backoff."
            )
        elif "rate limit" in sig.lower() or "429" in sig:
            proposals.append(
                f"For `{tool}`, add retry-with-jitter and circuit breaker around bursty phases."
            )
        else:
            proposals.append(
                f"Add precondition checks and better error surfacing for `{tool}` in planner graph."
            )
    return proposals


def write_patch_proposals(summary: dict[str, Any] | None = None) -> Path | None:
    summary = summary or cluster_failures()
    proposals = propose_resilience_patches(summary)
    if not proposals:
        return None
    out_dir = Path("docs/patches/auto_patches")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out_file = out_dir / f"resilience_proposals_{ts}.md"
    lines = ["# Automatic Resilience Patch Proposals", "", f"Generated: {ts}", "", "## Proposals:"]
    for p in proposals:
        lines.append(f"- {p}")
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file

