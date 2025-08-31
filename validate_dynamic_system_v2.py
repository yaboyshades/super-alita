#!/usr/bin/env python3
"""
Super Alita Dynamic System Validation Script (v2)
Uses the runtime's REUG tool routes for consensus validation.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict

import requests


BASE = "http://127.0.0.1:8080"


def test_health() -> bool:
    try:
        r = requests.get(f"{BASE}/healthz", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"? Health: {data.get('status','unknown')}")
            return True
        print(f"? Health failed: {r.status_code}")
        return False
    except Exception as e:
        print(f"? Health error: {e}")
        return False


def test_tools_catalog() -> Dict[str, Any]:
    try:
        r = requests.get(f"{BASE}/tools/catalog", timeout=5)
        if r.status_code != 200:
            print(f"? Catalog failed: {r.status_code}")
            return {"success": False}
        tools = r.json()
        names = [t.get("name", "unnamed") for t in tools]
        print(f"? Tools: {len(names)} available")
        print(
            f"   ?? First: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}"
        )
        dyn = [n for n in ["deepconf_consensus", "reug_start_turn", "reug_stream_next"] if n in names]
        print(f"   ?? Dynamic: {', '.join(dyn)}")
        return {"success": True, "tools": tools}
    except Exception as e:
        print(f"? Catalog error: {e}")
        return {"success": False}


def test_consensus_via_reug() -> bool:
    """Start a REUG turn and verify deepconf_consensus result appears in stream."""
    try:
        start_payload = {
            "message": (
                "Use deepconf_consensus with prompt 'What is machine learning?' "
                "and 2 samples and method weighted_vote"
            ),
            "session_id": "validation_consensus",
        }
        s = requests.post(f"{BASE}/tools/reug_start_turn", json=start_payload, timeout=10)
        if s.status_code != 200:
            print(f"? start_turn failed: {s.status_code}")
            print(f"   body: {s.text[:200]}")
            return False
        run_id = s.json().get("run_id")
        if not run_id:
            print("? no run_id from start_turn")
            return False

        found = False
        consensus_text = ""
        confidence = 0.0
        method = None
        for _ in range(24):  # ~24 polls
            n = requests.post(
                f"{BASE}/tools/reug_stream_next",
                json={"run_id": run_id},
                timeout=8,
            )
            if n.status_code != 200:
                print(f"? stream_next failed: {n.status_code}")
                print(f"   body: {n.text[:200]}")
                return False
            data = n.json()
            chunks = data.get("chunks", []) or []
            finished = data.get("finished", False)
            for ev in chunks:
                if isinstance(ev, dict) and ev.get("type") == "AbilitySucceeded" and ev.get("tool") == "deepconf_consensus":
                    res = ev.get("result", {}) or {}
                    consensus_text = res.get("consensus_text", "")
                    try:
                        confidence = float(res.get("consensus_confidence", 0) or 0)
                    except Exception:
                        confidence = 0.0
                    method = res.get("aggregation_method") or "weighted_vote"
                    found = True
                if isinstance(ev, dict) and ev.get("type") == "TaskSucceeded" and not consensus_text:
                    try:
                        consensus_text = (ev.get("data", {}) or {}).get("content", "")
                    except Exception:
                        pass
            if found or finished:
                break

        if found:
            print("? Consensus via REUG: OK")
            print(f"   ?? Confidence: {confidence:.2f}")
            if method:
                print(f"   ?? Method: {method}")
            print(f"   ?? Text: {consensus_text[:120]}{'...' if len(consensus_text) > 120 else ''}")
            return True
        print("? Consensus not observed in stream")
        return False
    except Exception as e:
        print(f"? Consensus error: {e}")
        return False


def test_streaming_chat() -> bool:
    try:
        payload = {"message": "Hello from validator", "session_id": "validation_test"}
        r = requests.post(f"{BASE}/v1/chat/stream", json=payload, timeout=10, stream=True)
        if r.status_code != 200:
            print(f"? Stream failed: {r.status_code}")
            return False
        first = next(r.iter_lines(decode_unicode=True), None)
        if first:
            print("? Streaming: OK")
            print(f"   ?? First: {str(first)[:80]}{'...' if first and len(str(first))>80 else ''}")
            return True
        print("? Streaming: no content")
        return False
    except Exception as e:
        print(f"? Streaming error: {e}")
        return False


def main() -> int:
    print("?? Super Alita Dynamic System Validation (v2)")
    print("=" * 54)

    tests: list[tuple[str, Any]] = [
        ("Health", test_health),
        ("Tools Catalog", lambda: test_tools_catalog()["success"]),
        ("Consensus via REUG", test_consensus_via_reug),
        ("Streaming Chat", test_streaming_chat),
    ]
    results: list[tuple[str, bool]] = []
    for name, fn in tests:
        print(f"\n?? {name}...")
        ok = False
        try:
            ok = fn()
        except Exception as e:
            print(f"? {name} exception: {e}")
        results.append((name, ok))

    print("\n" + "=" * 54)
    print("?? SUMMARY")
    print("=" * 54)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(("? PASS: " if ok else "? FAIL: ") + name)
    print(f"\n?? Overall: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

