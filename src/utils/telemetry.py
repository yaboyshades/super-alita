#!/usr/bin/env python3
from __future__ import annotations
import time
from typing import Dict, Any
from src.prompt.policy_constants import PROMPT_VERSION, SCHEMA_VERSION

class ReadLedger:
    def __init__(self):
        self._seen = set()
    def record(self, doc_id: str, chunk_idx: int):
        self._seen.add((doc_id, int(chunk_idx)))
    def already_read(self, doc_id: str, chunk_idx: int) -> bool:
        return (doc_id, int(chunk_idx)) in self._seen
    def count(self) -> int:
        return len(self._seen)

def start_timer() -> float:
    return time.time()

def end_timer(t0: float) -> int:
    return int((time.time() - t0) * 1000)

def telemetry_footer(retrieval_rounds: int, segments_used: int, totals: Dict[str,int]) -> Dict[str, Any]:
    return {
        "telemetry": {
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "retrieval_mode": "segmented",
            "retrieval_rounds": retrieval_rounds,
            "segments_used": segments_used,
            "total_extracted": totals
        }
    }
