#!/usr/bin/env python3
"""
Unified system coherence validator for Super Alita.

Covers end-to-end integration across major components:
- Health endpoints
- Tools catalog (static + dynamic)
- Direct ability execution (deepconf_consensus)
- Streaming chat (SSE frame presence)
- MCP brainstorm → register → execute loop
- GUI components reachability (mcp_index)

Environment variables:
- BASE_URL or SUPER_ALITA_BASE_URL (default: http://127.0.0.1:8080)
- VALIDATOR_TIMEOUT (per-request read timeout, default: 60s)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

BASE = os.getenv("SUPER_ALITA_BASE_URL", os.getenv("BASE_URL", "http://127.0.0.1:8080"))
READ_TIMEOUT = int(os.getenv("VALIDATOR_TIMEOUT", "60"))


def _ok(label: str) -> None:
    print(f"OK: {label}")


def _fail(label: str, msg: str) -> None:
    print(f"FAIL: {label}: {msg}")


def _get_json(url: str, timeout: int | float = READ_TIMEOUT) -> tuple[bool, Any]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        return True, r.json()
    except Exception as e:
        return False, str(e)


def _post_json(url: str, payload: dict[str, Any], timeout: int | float = READ_TIMEOUT) -> tuple[bool, Any]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return False, {"status": r.status_code, "text": r.text[:400]}
        return True, r.json()
    except Exception as e:
        return False, str(e)


def test_health() -> bool:
    ok, data = _get_json(f"{BASE}/healthz", timeout=10)
    if ok and isinstance(data, dict) and data.get("status") in {"healthy", "ok"}:
        _ok("Health")
        return True
    _fail("Health", str(data))
    return False


def test_catalog() -> tuple[bool, list[str]]:
    ok, data = _get_json(f"{BASE}/tools/catalog", timeout=15)
    if not ok:
        _fail("Tools catalog", str(data))
        return False, []
    try:
        names = [t.get("name") for t in data if isinstance(t, dict)]
        have = all(n in names for n in ["reug_start_turn", "reug_stream_next"])
        if have:
            _ok(f"Catalog ({len(names)} tools)")
            return True, names
        _fail("Catalog", f"missing required tools in {names[:10]}")
        return False, names
    except Exception as e:
        _fail("Catalog", str(e))
        return False, []


def test_ability_consensus() -> bool:
    payload = {
        "prompt": "In one sentence, what is 2+2?",
        "num_samples": 1,
        "temperature": 0.2,
        "max_tokens": 32,
    }
    ok, data = _post_json(f"{BASE}/ability/execute/deepconf_consensus", payload, timeout=120)
    if not ok:
        _fail("Ability: deepconf_consensus", json.dumps(data))
        return False
    try:
        res = data.get("result", {})
        ctext = res.get("consensus_text", "")
        conf = res.get("consensus_confidence", 0)
        transports = (res.get("metadata", {}) or {}).get("transports", {})
        assert isinstance(ctext, str)
        assert isinstance(conf, (int, float))
        _ok(f"Ability consensus (transport={json.dumps(transports)})")
        return True
    except Exception as e:
        _fail("Ability consensus parse", str(e))
        return False


def test_streaming() -> bool:
    try:
        r = requests.post(
            f"{BASE}/v1/chat/stream",
            json={"message": "Say hello", "session_id": "coherence"},
            timeout=15,
            stream=True,
        )
        if r.status_code != 200:
            _fail("Streaming", f"HTTP {r.status_code}")
            return False
        first = next(r.iter_lines(decode_unicode=True), None)
        if first:
            _ok("Streaming (SSE frame received)")
            return True
        _fail("Streaming", "no frames")
        return False
    except Exception as e:
        _fail("Streaming", str(e))
        return False


def test_mcp_loop() -> bool:
    # Brainstorm
    ok, props = _post_json(f"{BASE}/tools/mcp/brainstorm", {"task": "Fetch a URL and extract text"}, timeout=15)
    if not ok:
        _fail("MCP brainstorm", json.dumps(props))
        return False
    proposals = props.get("proposals", []) if isinstance(props, dict) else []
    if not proposals:
        _fail("MCP brainstorm", "no proposals")
        return False
    spec = proposals[0]
    # Register
    ok, reg = _post_json(f"{BASE}/tools/mcp/register", spec, timeout=15)
    if not ok:
        _fail("MCP register", json.dumps(reg))
        return False
    tool_id = reg.get("registered")
    if not tool_id:
        _fail("MCP register", "no registered id")
        return False
    # Execute
    ok, out = _post_json(
        f"{BASE}/tools/execute/{tool_id}", {"args": {"url": "https://example.com", "truncate": 500}}, timeout=20
    )
    if not ok:
        _fail("MCP execute", json.dumps(out))
        return False
    try:
        content = (out.get("result", {}) or {}).get("content", "")
        if isinstance(content, str) and ("Example Domain" in content or len(content) > 0):
            _ok(f"MCP loop ({tool_id})")
            return True
        _fail("MCP execute", "unexpected content")
        return False
    except Exception as e:
        _fail("MCP execute parse", str(e))
        return False


def test_gui_mcp_index() -> bool:
    try:
        r = requests.get(f"{BASE}/gui/components/mcp_index", timeout=10)
        if r.status_code == 200 and "MCP Box" in r.text:
            _ok("GUI mcp_index")
            return True
        _fail("GUI mcp_index", f"HTTP {r.status_code}")
        return False
    except Exception as e:
        _fail("GUI mcp_index", str(e))
        return False


def main() -> int:
    print("Super Alita - System Coherence Validation")
    print("=" * 56)
    steps = [
        ("health", test_health),
        ("catalog", lambda: test_catalog()[0]),
        ("ability_consensus", test_ability_consensus),
        ("streaming", test_streaming),
        ("mcp_loop", test_mcp_loop),
        ("gui_mcp_index", test_gui_mcp_index),
    ]

    passed = 0
    for name, fn in steps:
        try:
            ok = fn()
            passed += 1 if ok else 0
        except Exception as e:
            _fail(name, str(e))

    print("\n" + "=" * 56)
    print(f"Summary: {passed}/{len(steps)} checks passed")
    report = {
        "passed": passed,
        "total": len(steps),
        "base_url": BASE,
        "timestamp": int(time.time()),
    }
    try:
        with open("coherence_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Report: coherence_report.json")
    except Exception:
        pass
    return 0 if passed == len(steps) else 1


if __name__ == "__main__":
    sys.exit(main())
