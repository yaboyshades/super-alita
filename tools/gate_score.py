#!/usr/bin/env python3
"""
Gate Scorer

Scores a requirement against the Coding Gate rubric and appends a row to
99_gate_log.csv. Optionally updates 10_requirements_ledger.csv status.

Usage:
  python tools/gate_score.py --req-id REQ-001 [--auto] [--score 90] \
      [--status pass|fail] [--notes "..."] [--update-ledger]

Heuristics (when --auto is used):
  - If any test file mentions the req id → +25 (tests present)
  - If status in ledger is in_review/complete → +10 (spec compliance proxy)
  - Default baseline → 60
  - Pass threshold: 85
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
LEDGER = ROOT / "10_requirements_ledger.csv"
GATE_LOG = ROOT / "99_gate_log.csv"
TESTS_DIR = ROOT / "tests"


def read_ledger() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not LEDGER.exists():
        return rows
    with LEDGER.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(
            line for line in f if not line.lstrip().startswith("#")
        )
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def write_ledger(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with LEDGER.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_gate_log(log_id: str, req_id: str, score: int, status: str, notes: str, reviewer: str) -> None:
    header = ["log_id", "timestamp", "req_id", "score", "status", "notes", "reviewer"]
    rows_exist = GATE_LOG.exists() and GATE_LOG.stat().st_size > 0
    with GATE_LOG.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not rows_exist:
            writer.writerow(["# ---"])  # lightweight header separators
            writer.writerow(header)
        ts = datetime.now(UTC).isoformat()
        writer.writerow([log_id, ts, req_id, score, status, notes, reviewer])


def auto_score(req_id: str) -> int:
    score = 60
    # Tests mention
    if TESTS_DIR.exists():
        for p in TESTS_DIR.rglob("test_*.py"):
            try:
                if req_id in p.read_text(encoding="utf-8", errors="ignore"):
                    score += 25
                    break
            except Exception:
                continue
    # Ledger status proxy
    for row in read_ledger():
        if row.get("req_id") == req_id:
            if row.get("status", "").lower() in {"in_review", "complete"}:
                score += 10
            break
    return min(score, 100)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate scorer")
    ap.add_argument("--req-id", required=True, help="Requirement ID, e.g., REQ-001")
    ap.add_argument("--auto", action="store_true", help="Compute score heuristically")
    ap.add_argument("--score", type=int, help="Override score (0-100)")
    ap.add_argument("--status", choices=["pass", "fail"], help="Gate result status")
    ap.add_argument("--notes", default="", help="Notes to store in gate log")
    ap.add_argument("--reviewer", default="system", help="Reviewer name")
    ap.add_argument("--update-ledger", action="store_true", help="Update ledger status on pass")
    args = ap.parse_args()

    if not LEDGER.exists():
        print(f"❌ Ledger not found at {LEDGER}")
        return 1

    # Choose score
    score = args.score if args.score is not None else (auto_score(args.req_id) if args.auto else 60)
    status = args.status or ("pass" if score >= 85 else "fail")
    log_id = f"LOG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    append_gate_log(log_id, args.req_id, score, status, args.notes, args.reviewer)
    print(f"📝 Gate logged: {log_id} score={score} status={status}")

    if args.update_ledger and status == "pass":
        rows = read_ledger()
        updated = False
        for r in rows:
            if r.get("req_id") == args.req_id:
                if r.get("status", "").lower() != "complete":
                    r["status"] = "complete"
                    updated = True
                break
        if updated:
            write_ledger(rows)
            print(f"✅ Ledger updated: {args.req_id} → complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

